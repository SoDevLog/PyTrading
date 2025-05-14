# Big Error encountered several times with 'pymeeus'

When strated to install from zero a new python's environnement I found no way to get pymeeus :

"\.venv_3_9\Lib\site-packages\pymeeus"

It'always empty.

So a create a test script: test_workalendar.py

Otherwize I'll get this error.

    (.venv_3_9) ps c:\users\bruno\documents\github\pythonadvanced\z-tests> python test_workalendar.py                                                                  
    traceback (most recent call last):
    file "c:\users\bruno\documents\github\pythonadvanced\z-tests\test_workalendar.py", line 6, in <module>
        from workalendar.europe import france
    file "c:\users\bruno\documents\github\pythonadvanced\.venv_3_9\lib\site-packages\workalendar\europe\__init__.py", line 1, in <module>
        from .austria import austria
    file "c:\users\bruno\documents\github\pythonadvanced\.venv_3_9\lib\site-packages\workalendar\europe\austria.py", line 1, in <module>
        from ..core import westerncalendar
    file "c:\users\bruno\documents\github\pythonadvanced\.venv_3_9\lib\site-packages\workalendar\core.py", line 11, in <module>
        import convertdate
    file "c:\users\bruno\documents\github\pythonadvanced\.venv_3_9\lib\site-packages\convertdate\__init__.py", line 26, in <module>
        from . import french_republican
    file "c:\users\bruno\documents\github\pythonadvanced\.venv_3_9\lib\site-packages\convertdate\french_republican.py", line 45, in <module>
        from pymeeus.sun import sun
    modulenotfounderror: no module named 'pymeeus'
    (.venv_3_9) ps c:\users\bruno\documents\github\pythonadvanced\z-tests>

From days to days I just found an install that works and copy :
"\.venv_3_9\Lib\site-packages\pymeeus"
"\.venv_3_9\Lib\site-packages\pymeeus-0.5.12.dist-info"

Into my new install, funking shit!

I just hope next time I'll remenber where I put this ;-))
