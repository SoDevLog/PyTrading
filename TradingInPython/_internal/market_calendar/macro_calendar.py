# -*- coding: utf-8 -*-
""" Générateur autonome de calendar-macro.json pour TradingInPython.

    Sources structurées uniquement :
        - FRED : API JSON officielle pour les publications américaines.
        - Eurostat : flux iCalendar officiel des Euro-indicateurs.
        - BCE / BoE / BoJ : entrées officielles MANUAL conservées dans le JSON ;
        des flux iCalendar peuvent être configurés si une URL officielle est disponible.

    Aucune page HTML n'est téléchargée ni analysée.
    Les entrées MANUAL/CUSTOM présentes dans calendar-macro.json sont préservées
    lors de chaque régénération, dans la fenêtre demandée.

    Configuration dans BASE_CONFIG_DIR/.env :
        FRED_API_KEY=...
        EUROSTAT_ICS_URL=...       (recommandé pour compléter les indicateurs européens)
        ECB_ICS_URL=...            (facultatif, si un flux iCalendar officiel est disponible)
        BOE_ICS_URL=...            (facultatif)
        BOJ_ICS_URL=...            (facultatif)

    Exécution depuis la racine du projet :
        >python .\\market_calendar\\macro_calendar.py --months 12
        
"""
from __future__ import annotations

import argparse
import calendar
import getpass
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.path import BASE_CONFIG_DIR

try:
    from dotenv import load_dotenv
    load_dotenv(BASE_CONFIG_DIR / '.env')
except ImportError:
    pass

FRED_DATES_URL = 'https://api.stlouisfed.org/fred/release/dates'
USER_AGENT = 'TradingInPython-MacroCalendar/3.0'
HTTP_TIMEOUT = 25

# release_id: (libellé affiché, catégorie, importance 1..3)
FRED_RELEASES: dict[int, tuple[str, str, int]] = {
    9:   ('Ventes au détail US', 'CONSOMMATION', 2),
    10:  ('CPI (US)', 'INFLATION', 3),
    11:  ("Coût de l'emploi US (Employment Cost Index)", 'EMPLOI', 2),
    13:  ('Production industrielle US', 'ACTIVITE', 2),
    26:  ('ISM manufacturier US', 'ACTIVITE', 3),
    46:  ('PPI US — prix à la production', 'INFLATION', 2),
    50:  ('Emploi US (NFP)', 'EMPLOI', 3),
    51:  ('Commerce extérieur US', 'COMMERCE', 2),
    53:  ('PIB US', 'CROISSANCE', 3),
    54:  ('PCE US — dépenses et inflation', 'INFLATION', 3),
    101: ('FOMC — communiqué de politique monétaire', 'BANQUE_CENTRALE', 3),
}

EUROSTAT_MACRO_RE = re.compile(
    r'(\b(hicp|cpi|consumer price|inflation|gdp|gross domestic|unemployment|employment|job vacancy|'
    r'industrial production|industrial producer prices|producer prices|retail trade|'
    r'economic sentiment|business and consumer survey|trade in goods|balance of payments|'
    r'government debt|government deficit|labour cost|labor cost|wage|construction|'
    r'services production|industrial import prices|house price index|interest rates)\b|'
    r'prix à la consommation|indice des prix|inflation|produit intérieur brut|\bPIB\b|'
    r'chômage|emploi|coût du travail|production industrielle|prix à la production|'
    r'commerce de détail|confiance économique|commerce de biens|balance des paiements|'
    r'dette publique|déficit public|salaires|production des services|permis de construire)',
    re.IGNORECASE,
)
CENTRAL_BANK_RE = re.compile(
    r'(monetary policy|policy meeting|rate decision|interest rate|governing council|'
    r'monetary policy committee|\bmpc\b|\bfomc\b|tankan|bank rate|policy board)',
    re.IGNORECASE,
)

# Un flux optionnel peut être ajouté dans .env sans changer le code.
FEED_CONFIG = (
    ('EUROSTAT', 'EUROSTAT_ICS_URL', 'EU', 'EU', 'UE — ', EUROSTAT_MACRO_RE, 'ACTIVITE'),
    ('ECB', 'ECB_ICS_URL', 'EU', 'EA', 'BCE — ', CENTRAL_BANK_RE, 'BANQUE_CENTRALE'),
    ('BOE', 'BOE_ICS_URL', 'UK', 'GB', 'BoE — ', CENTRAL_BANK_RE, 'BANQUE_CENTRALE'),
    ('BOJ', 'BOJ_ICS_URL', 'ASIA', 'JP', 'BoJ — ', CENTRAL_BANK_RE, 'BANQUE_CENTRALE'),
)


def add_months(day: date, months: int) -> date:
    """Ajoute des mois calendaires en conservant le dernier jour valide."""
    month_index = day.month - 1 + months
    year = day.year + month_index // 12
    month = month_index % 12 + 1
    last_day = calendar.monthrange(year, month)[1]
    return date(year, month, min(day.day, last_day))


def _http_get(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={'User-Agent': USER_AGENT, 'Accept': 'application/json,text/calendar,*/*'},
    )
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
        return response.read()


def _event(
    day: date,
    label: str,
    *,
    category: str,
    importance: int,
    source: str,
    region: str,
    country: str,
    status: str = 'programmé',
    url: str = '',
) -> dict[str, Any]:
    item: dict[str, Any] = {
        'date': day.isoformat(),
        'label': label.strip(),
        'category': category,
        'region': region,
        'country': country,
        'importance': max(1, min(3, int(importance))),
        'source': source,
        'status': status,
    }
    if url:
        item['url'] = url
    return item

# -----------------------------------------------------------------------------

def fetch_fred(api_key: str, start: date, end: date) -> tuple[list[dict[str, Any]], list[str]]:
    """Récupère les dates FRED futures via son API JSON."""
    events: list[dict[str, Any]] = []
    warnings: list[str] = []

    for index, (release_id, (label, category, importance)) in enumerate(FRED_RELEASES.items()):
        # Espacement prudent des appels pour éviter une limite de débit.
        if index:
            time.sleep(0.55)
        params = urllib.parse.urlencode({
            'release_id': release_id,
            'api_key': api_key,
            'file_type': 'json',
            'sort_order': 'asc',
            'limit': 10000,
            'include_release_dates_with_no_data': 'true',
        })
        url = f'{FRED_DATES_URL}?{params}'
        try:
            payload = json.loads(_http_get(url).decode('utf-8'))
            if 'error_code' in payload or 'error_message' in payload:
                raise RuntimeError(payload.get('error_message', 'réponse API FRED invalide'))
            for row in payload.get('release_dates', []):
                try:
                    day = date.fromisoformat(str(row.get('date', ''))[:10])
                except ValueError:
                    continue
                if start <= day <= end:
                    events.append(_event(
                        day, label, category=category, importance=importance,
                        source=f'FRED:{release_id}', region='US', country='US',
                        status='programmé',
                        url=f'https://fred.stlouisfed.org/release?rid={release_id}',
                    ))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError, OSError) as exc:
            warnings.append(f'FRED {release_id} ({label}) : {exc}')

    if not events:
        raise RuntimeError(
            "Aucune date FRED récupérée. Vérifie la clé API et la connexion ; "
            "le fichier existant n'a pas été modifié."
        )
    return events, warnings

# -----------------------------------------------------------------------------

def _unfold_ics(text: str) -> list[str]:
    """Déplie les lignes iCalendar repliées (RFC 5545)."""
    lines: list[str] = []
    for line in text.replace('\r\n', '\n').replace('\r', '\n').split('\n'):
        if line.startswith((' ', '\t')) and lines:
            lines[-1] += line[1:]
        else:
            lines.append(line)
    return lines


def _ics_unescape(value: str) -> str:
    return (value.replace(r'\n', ' ').replace(r'\N', ' ')
            .replace(r'\,', ',').replace(r'\;', ';')
            .replace(r'\\', '\\').strip())


def _ics_date(value: str) -> date | None:
    """Lit DATE ou DATE-TIME iCalendar en conservant la date locale du flux."""
    match = re.match(r'^(\d{4})(\d{2})(\d{2})', value.strip())
    if not match:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def _classify(label: str, default: str = 'ACTIVITE') -> tuple[str, int]:
    low = label.casefold()
    if any(x in low for x in ('inflation', 'hicp', 'cpi', 'consumer price', 'producer price', 'ppi')):
        return 'INFLATION', 3
    if any(x in low for x in ('gdp', 'gross domestic', 'pib')):
        return 'CROISSANCE', 3
    if any(x in low for x in ('unemployment', 'employment', 'payroll', 'nfp', 'wage', 'labour cost')):
        return 'EMPLOI', 3
    if any(x in low for x in ('interest rate', 'rate decision', 'monetary policy', 'bank rate', 'mpc', 'fomc')):
        return 'BANQUE_CENTRALE', 3
    if any(x in low for x in ('retail', 'trade', 'consumption', 'sales')):
        return 'CONSOMMATION', 2
    return default, 2

# -----------------------------------------------------------------------------

def parse_ics(
    text: str,
    start: date,
    end: date,
    *,
    source: str,
    region: str,
    country: str,
    prefix: str = '',
    include_re: re.Pattern[str] | None = None,
    default_category: str = 'ACTIVITE',
) -> list[dict[str, Any]]:
    """Convertit un flux iCalendar en événements normalisés, sans analyser de HTML."""
    events: list[dict[str, Any]] = []
    current: dict[str, str] | None = None

    for line in _unfold_ics(text):
        upper = line.upper()
        if upper == 'BEGIN:VEVENT':
            current = {}
            continue
        if upper == 'END:VEVENT':
            if current:
                day = _ics_date(current.get('DTSTART', ''))
                title = _ics_unescape(current.get('SUMMARY', ''))
                state = current.get('STATUS', '').upper()
                if day and title and start <= day <= end and state != 'CANCELLED':
                    if include_re is None or include_re.search(title):
                        category, importance = _classify(title, default_category)
                        status = 'confirmé' if state == 'CONFIRMED' else 'provisoire' if state == 'TENTATIVE' else 'programmé'
                        item_url = _ics_unescape(current.get('URL', ''))
                        events.append(_event(
                            day, f'{prefix}{title}', category=category,
                            importance=importance, source=source, region=region,
                            country=country, status=status, url=item_url,
                        ))
            current = None
            continue
        if current is None or ':' not in line:
            continue
        name, value = line.split(':', 1)
        key = name.split(';', 1)[0].upper()
        if key in ('DTSTART', 'SUMMARY', 'STATUS', 'URL'):
            current[key] = value.strip()
    return events

# -----------------------------------------------------------------------------

def fetch_ics_feed(
    url: str,
    start: date,
    end: date,
    *,
    source: str,
    region: str,
    country: str,
    prefix: str,
    include_re: re.Pattern[str] | None,
    default_category: str,
) -> list[dict[str, Any]]:
    raw = _http_get(url).decode('utf-8-sig', errors='replace')
    if 'BEGIN:VCALENDAR' not in raw.upper():
        raise ValueError('la réponse ne ressemble pas à un flux iCalendar (BEGIN:VCALENDAR absent)')
    return parse_ics(
        raw, start, end, source=source, region=region, country=country,
        prefix=prefix, include_re=include_re, default_category=default_category,
    )


def _load_manual_events(path: Path, start: date, end: date) -> list[dict[str, Any]]:
    """Préserve les entrées MANUAL/CUSTOM et les anciennes entrées sans source."""
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
        items = payload.get('events', []) if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            return []
        manual: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict) or not item.get('date') or not item.get('label'):
                continue
            try:
                item_day = date.fromisoformat(str(item['date'])[:10])
            except ValueError:
                continue
            if not start <= item_day <= end:
                continue
            source = str(item.get('source', '')).upper()
            if source in ('MANUAL', 'CUSTOM') or source.startswith(('MANUAL:', 'CUSTOM:')) or not source:
                preserved = dict(item)
                preserved.setdefault('source', 'MANUAL')
                preserved.setdefault('status', 'confirmé')
                preserved.setdefault('importance', 2)
                manual.append(preserved)
        return manual
    except (OSError, json.JSONDecodeError):
        return []


def deduplicate_and_sort(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Déduplique par date/libellé ; une entrée manuelle est prioritaire."""
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for item in events:
        key = (str(item.get('date', '')), str(item.get('label', '')).casefold())
        if not all(key):
            continue
        previous = unique.get(key)
        source = str(item.get('source', '')).upper()
        is_manual = source in ('MANUAL', 'CUSTOM') or source.startswith(('MANUAL:', 'CUSTOM:'))
        if previous is None or is_manual:
            unique[key] = item

    def sort_key(item: dict[str, Any]):
        try:
            importance = int(item.get('importance', 2))
        except (TypeError, ValueError):
            importance = 2
        return str(item['date']), -importance, str(item['label']).casefold()

    return sorted(unique.values(), key=sort_key)


def write_json(path: Path, events: list[dict[str, Any]]) -> None:
    """Écriture atomique : le fichier précédent reste intact si l'écriture échoue."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(events, stream, ensure_ascii=False, indent=4)
        stream.write('\n')
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description='Génère calendar-macro.json pour TradingInPython')
    parser.add_argument('--months', type=int, default=12, help='Horizon en mois (défaut : 12)')
    parser.add_argument('--output', type=Path, default=BASE_CONFIG_DIR / 'calendar-macro.json')
    parser.add_argument('--skip-eurostat', action='store_true', help='Ne pas interroger le flux iCalendar Eurostat')
    args = parser.parse_args()
    if not 1 <= args.months <= 36:
        parser.error('--months doit être compris entre 1 et 36')

    api_key = os.getenv('FRED_API_KEY', '').strip()
    if not api_key:
        api_key = getpass.getpass('Clé API FRED (saisie masquée) : ').strip()
    if not api_key:
        print('ERREUR : clé FRED absente ; fichier non modifié.', file=sys.stderr)
        return 2

    today = date.today()
    end = add_months(today, args.months)
    output = args.output
    print(f'Période : {today:%d/%m/%Y} → {end:%d/%m/%Y} ({args.months} mois)')

    print('[1/2] FRED — publications américaines via API JSON')
    try:
        fred_events, fred_warnings = fetch_fred(api_key, today, end)
        if fred_warnings:
            for warning in fred_warnings:
                print(f'AVERTISSEMENT : {warning}', file=sys.stderr)
            print('Mise à jour annulée : au moins une série FRED n’a pas pu être vérifiée.', file=sys.stderr)
            print('Le calendar-macro.json existant n’a pas été modifié.', file=sys.stderr)
            return 5
    except Exception as exc:
        print(f'ERREUR FRED : {exc}', file=sys.stderr)
        print('Le calendar-macro.json existant n’a pas été modifié.', file=sys.stderr)
        return 3

    events: list[dict[str, Any]] = list(fred_events)
    warnings: list[str] = []
    print(f'      {len(fred_events)} événement(s) FRED')

    print('[2/2] Flux iCalendar officiels configurés')
    configured = 0
    feed_errors: list[str] = []
    for source, env_name, region, country, prefix, pattern, default_category in FEED_CONFIG:
        if args.skip_eurostat and source == 'EUROSTAT':
            continue
        feed_url = os.getenv(env_name, '').strip()
        if not feed_url:
            continue
        configured += 1
        try:
            found = fetch_ics_feed(
                feed_url, today, end, source=source, region=region, country=country,
                prefix=prefix, include_re=pattern, default_category=default_category,
            )
            events.extend(found)
            print(f'      {source} : {len(found)} événement(s)')
            if not found:
                warnings.append(f'{source} : aucun événement retenu dans la période ; vérifier le flux et son filtre.')
        except Exception as exc:
            feed_errors.append(f'{source} : {exc}')

    # Ne pas écraser le dernier calendrier valide si un flux explicitement configuré échoue.
    if feed_errors:
        print('Mise à jour annulée : échec d’au moins un flux iCalendar configuré.', file=sys.stderr)
        for error in feed_errors:
            print(f'  - {error}', file=sys.stderr)
        print('Le calendar-macro.json existant n’a pas été modifié.', file=sys.stderr)
        return 6

    if not args.skip_eurostat and not os.getenv('EUROSTAT_ICS_URL', '').strip():
        warnings.append('Eurostat non configuré : ajoute EUROSTAT_ICS_URL dans BASE_CONFIG_DIR/.env pour les publications macro européennes.')

    manual_events = _load_manual_events(output, today, end)
    events.extend(manual_events)
    events = deduplicate_and_sort(events)
    if not events:
        print('ERREUR : aucun événement ; fichier non modifié.', file=sys.stderr)
        return 4

    write_json(output, events)
    print(f'\nOK : {len(events)} événement(s) écrits dans {output}')
    by_source: dict[str, int] = {}
    for item in events:
        key = str(item.get('source', 'INCONNUE')).split(':', 1)[0]
        by_source[key] = by_source.get(key, 0) + 1
    print('Sources : ' + ' | '.join(f'{name}: {count}' for name, count in sorted(by_source.items())))

    manual_sources = {str(item.get('source', '')).upper() for item in manual_events}
    for central_bank, source_name in (('BCE', 'MANUAL:ECB'), ('BoE', 'MANUAL:BOE'), ('BoJ', 'MANUAL:BOJ')):
        feed_name = {'BCE': 'ECB', 'BoE': 'BOE', 'BoJ': 'BOJ'}[central_bank]
        if source_name not in manual_sources and not os.getenv(f'{feed_name}_ICS_URL', '').strip():
            warnings.append(f'{central_bank} : aucun flux iCalendar configuré ni événement manuel conservé.')

    if configured == 0:
        print('Note : aucun flux iCalendar configuré. Les dates manuelles BCE/BoE/BoJ sont conservées si elles existent déjà dans le JSON.')
    if warnings:
        print('\nAVERTISSEMENTS :')
        for warning in warnings:
            print(f'  - {warning}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
