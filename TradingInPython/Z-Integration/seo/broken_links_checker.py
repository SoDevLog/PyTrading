#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# >python broken_links_checker.py https://www.trading-et-data-analyses.com/p/outil-de-prediction-avec-keras-et.html
# >python broken_links_checker.py https://www.trading-et-data-analyses.com/p/outil-de-prediction-avec-keras-et.html --domain-only
#
# --domain-only : les liens externes comment https://facebook.com, https://google.com sont ignorés ❌
#

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Désactiver les warnings SSL pour éviter le spam
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class BrokenLinksChecker:
    def __init__(self, timeout=15, max_workers=10, delay=0.5):
        self.timeout = timeout
        self.max_workers = max_workers
        self.delay = delay
        self.session = requests.Session()
        
        # Configuration des headers plus réalistes
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'fr-FR,fr;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        # Configuration de la stratégie de retry
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],  # Changé de method_whitelist
            backoff_factor=1
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def get_html_content(self, url):
        """Récupère le contenu HTML d'une URL"""
        try:
            if url.startswith('http'):
                response = self.session.get(url, timeout=self.timeout, verify=False)
                response.raise_for_status()
                return response.text
            else:
                # Lecture d'un fichier local
                with open(url, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            print(f"Erreur lors de la récupération de {url}: {e}")
            return None
    
    def extract_links(self, html_content, base_url=None):
        """Extrait tous les liens d'un contenu HTML"""
        soup = BeautifulSoup(html_content, 'html.parser')
        links = []
        
        # Récupération des liens <a href="">
        for link in soup.find_all('a', href=True):
            href = link['href'].strip()
            if href and not href.startswith('#') and not href.startswith('mailto:') and not href.startswith('tel:'):
                if base_url:
                    full_url = urljoin(base_url, href)
                else:
                    full_url = href
                links.append({
                    'url': full_url,
                    'text': link.get_text(strip=True)[:50] + '...' if len(link.get_text(strip=True)) > 50 else link.get_text(strip=True),
                    'type': 'link'
                })
        
        # Récupération des images <img src="">
        for img in soup.find_all('img', src=True):
            src = img['src'].strip()
            if src:
                if base_url:
                    full_url = urljoin(base_url, src)
                else:
                    full_url = src
                links.append({
                    'url': full_url,
                    'text': img.get('alt', 'Image sans alt'),
                    'type': 'image'
                })
        
        return links
    
    def is_problematic_domain(self, url):
        """Vérifie si le domaine est connu pour poser problème"""
        problematic_domains = [
            'facebook.com',
            'twitter.com',
            'linkedin.com',
            'instagram.com',
            'youtube.com',
            'google.com',
            'amazon.com'
        ]
        
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        
        return any(prob_domain in domain for prob_domain in problematic_domains)
    
    def check_link(self, link_info):
        """Vérifie si un lien est accessible"""
        url = link_info['url']
        
        try:
            # Ne pas vérifier les liens locaux (file://)
            if url.startswith('file://'):
                return {**link_info, 'status': 'local', 'status_code': None, 'error': None}
            
            # Vérifier seulement les liens HTTP/HTTPS
            if not url.startswith(('http://', 'https://')):
                return {**link_info, 'status': 'skipped', 'status_code': None, 'error': 'Protocole non supporté'}
            
            # Vérifier si c'est un domaine problématique
            if self.is_problematic_domain(url):
                return {**link_info, 'status': 'skipped', 'status_code': None, 'error': 'Domaine problématique (anti-bot)'}
            
            # Première tentative avec HEAD
            try:
                response = self.session.head(url, timeout=self.timeout, allow_redirects=True, verify=False)
                
                # Si HEAD ne fonctionne pas, essayer GET
                if response.status_code == 405 or response.status_code == 404:
                    response = self.session.get(url, timeout=self.timeout, allow_redirects=True, verify=False, stream=True)
                    # Fermer la connexion après les headers
                    if hasattr(response, 'close'):
                        response.close()
                
            except requests.exceptions.RequestException:
                # Si HEAD échoue, essayer GET
                response = self.session.get(url, timeout=self.timeout, allow_redirects=True, verify=False, stream=True)
                if hasattr(response, 'close'):
                    response.close()
            
            # Évaluation du statut
            if response.status_code < 400:
                status = 'ok'
            elif response.status_code == 403:
                # 403 peut être dû à un blocage anti-bot, pas forcément un lien cassé
                status = 'warning'
            elif response.status_code == 429:
                # Rate limiting
                status = 'warning'
            else:
                status = 'broken'
            
            return {
                **link_info,
                'status': status,
                'status_code': response.status_code,
                'error': None
            }
            
        except requests.exceptions.Timeout:
            return {
                **link_info,
                'status': 'timeout',
                'status_code': None,
                'error': 'Timeout'
            }
        except requests.exceptions.ConnectionError as e:
            return {
                **link_info,
                'status': 'connection_error',
                'status_code': None,
                'error': f'Erreur de connexion: {str(e)}'
            }
        except requests.exceptions.SSLError as e:
            return {
                **link_info,
                'status': 'ssl_error',
                'status_code': None,
                'error': f'Erreur SSL: {str(e)}'
            }
        except requests.exceptions.RequestException as e:
            return {
                **link_info,
                'status': 'error',
                'status_code': None,
                'error': str(e)
            }
        except Exception as e:
            return {
                **link_info,
                'status': 'error',
                'status_code': None,
                'error': str(e)
            }
    
    def check_all_links(self, url_or_file, filter_domain=None):
        """Vérifie tous les liens d'une page"""
        print(f"🔍 Analyse de: {url_or_file}")
        
        # Récupération du contenu HTML
        html_content = self.get_html_content(url_or_file)
        if not html_content:
            return []
        
        # Détermination de l'URL de base
        base_url = url_or_file if url_or_file.startswith('http') else None
        
        # Extraction des liens
        all_links = self.extract_links(html_content, base_url)
        
        # Filtrage par domaine si demandé
        if filter_domain:
            links = []
            for link in all_links:
                parsed_url = urlparse(link['url'])
                if parsed_url.netloc == '' or filter_domain in parsed_url.netloc:
                    links.append(link)
            print(f"📊 {len(links)} liens trouvés (filtrés pour {filter_domain})")
            print(f"📊 {len(all_links)} liens total (avant filtrage)")
        else:
            links = all_links
            print(f"📊 {len(links)} liens trouvés")
        
        if not links:
            return []
        
        # Vérification des liens avec threading
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_link = {executor.submit(self.check_link, link): link for link in links}
            
            for i, future in enumerate(as_completed(future_to_link), 1):
                result = future.result()
                results.append(result)
                
                # Affichage du progrès
                print(f"\r⏳ Progression: {i}/{len(links)} liens vérifiés", end='', flush=True)
                
                # Délai entre les requêtes
                if self.delay > 0:
                    time.sleep(self.delay)
        
        print()  # Nouvelle ligne après la barre de progression
        return results
    
    def print_results(self, results):
        """Affiche les résultats de la vérification"""
        if not results:
            print("❌ Aucun lien à vérifier")
            return
        
        broken_links = [r for r in results if r['status'] == 'broken']
        ok_links = [r for r in results if r['status'] == 'ok']
        warning_links = [r for r in results if r['status'] == 'warning']
        skipped_links = [r for r in results if r['status'] in ['local', 'skipped']]
        timeout_links = [r for r in results if r['status'] == 'timeout']
        connection_error_links = [r for r in results if r['status'] == 'connection_error']
        ssl_error_links = [r for r in results if r['status'] == 'ssl_error']
        error_links = [r for r in results if r['status'] == 'error']
        
        print(f"\n📈 RÉSUMÉ:")
        print(f"✅ Liens valides: {len(ok_links)}")
        print(f"❌ Liens cassés: {len(broken_links)}")
        print(f"⚠️  Avertissements (403/429): {len(warning_links)}")
        print(f"🕐 Timeouts: {len(timeout_links)}")
        print(f"🔌 Erreurs de connexion: {len(connection_error_links)}")
        print(f"🔒 Erreurs SSL: {len(ssl_error_links)}")
        print(f"⏭️  Liens ignorés: {len(skipped_links)}")
        print(f"🔥 Autres erreurs: {len(error_links)}")
        
        # Affichage des vraies erreurs en premier
        if broken_links:
            print(f"\n❌ LIENS VRAIMENT CASSÉS ({len(broken_links)}):")
            for link in broken_links:
                status_info = f"[{link['status_code']}]" if link['status_code'] else f"[{link['error']}]"
                print(f"  ❌ {status_info} {link['url']}")
                if link['text']:
                    print(f"     📝 Texte: {link['text']}")
                print()
        
        if warning_links:
            print(f"\n⚠️  AVERTISSEMENTS - À VÉRIFIER MANUELLEMENT ({len(warning_links)}):")
            for link in warning_links:
                status_info = f"[{link['status_code']}]" if link['status_code'] else f"[{link['error']}]"
                print(f"  ⚠️  {status_info} {link['url']}")
                if link['text']:
                    print(f"     📝 Texte: {link['text']}")
                print()
        
        if timeout_links:
            print(f"\n🕐 TIMEOUTS - SERVEUR LENT ({len(timeout_links)}):")
            for link in timeout_links:
                print(f"  🕐 {link['url']}")
                if link['text']:
                    print(f"     📝 Texte: {link['text']}")
                print()
        
        if connection_error_links:
            print(f"\n🔌 ERREURS DE CONNEXION ({len(connection_error_links)}):")
            for link in connection_error_links:
                print(f"  🔌 {link['url']}")
                print(f"     🔥 Erreur: {link['error']}")
                if link['text']:
                    print(f"     📝 Texte: {link['text']}")
                print()
        
        if ssl_error_links:
            print(f"\n🔒 ERREURS SSL ({len(ssl_error_links)}):")
            for link in ssl_error_links:
                print(f"  🔒 {link['url']}")
                print(f"     🔥 Erreur: {link['error']}")
                if link['text']:
                    print(f"     📝 Texte: {link['text']}")
                print()
        
        if error_links:
            print(f"\n🔥 AUTRES ERREURS ({len(error_links)}):")
            for link in error_links:
                print(f"  🔥 {link['url']}")
                print(f"     🔥 Erreur: {link['error']}")
                if link['text']:
                    print(f"     📝 Texte: {link['text']}")
                print()

def main():
    """Fonction principale"""
    if len(sys.argv) < 2:
        print("Usage: python broken_links_checker.py <URL_ou_fichier_HTML> [--domain-only]")
        print("Exemples:")
        print("  python broken_links_checker.py https://example.com")
        print("  python broken_links_checker.py https://example.com --domain-only")
        print("  python broken_links_checker.py index.html")
        sys.exit(1)
    
    url_or_file = sys.argv[1]
    domain_only = '--domain-only' in sys.argv
    
    # Configuration du vérificateur
    checker = BrokenLinksChecker(
        timeout=15,      # Délai d'attente en secondes (augmenté)
        max_workers=3,   # Nombre de threads simultanés (réduit pour éviter les blocages)
        delay=1.0        # Délai entre les requêtes (augmenté)
    )
    
    try:
        # Détermination du domaine à filtrer si nécessaire
        filter_domain = None
        if domain_only and url_or_file.startswith('http'):
            parsed_url = urlparse(url_or_file)
            filter_domain = parsed_url.netloc
            print(f"🎯 Mode analyse du domaine uniquement: {filter_domain}")
        
        # Vérification des liens
        results = checker.check_all_links(url_or_file, filter_domain)
        
        # Affichage des résultats
        checker.print_results(results)
        
        # Code de sortie basé sur les résultats (seulement les vrais liens cassés)
        broken_count = len([r for r in results if r['status'] == 'broken'])
        if broken_count > 0:
            print(f"\n❌ {broken_count} lien(s) vraiment cassé(s) détecté(s)")
            sys.exit(1)
        else:
            print(f"\n✅ Aucun lien cassé détecté!")
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Arrêt demandé par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n🔥 Erreur inattendue: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()