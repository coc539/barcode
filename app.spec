# app.spec

import os

from PyInstaller.utils.hooks import collect_all
 
datas, binaries, hiddenimports = [], [], []

for item in ['pyzbar', 'cv2', 'pytesseract']:

    tmp_ret = collect_all(item)

    datas.extend(tmp_ret[0])

    binaries.extend(tmp_ret[1])

    hiddenimports.extend(tmp_ret[2])
 
a = Analysis(

    ['test.py'],

    pathex=[],

    binaries=binaries,

    datas=datas + [

        ('C:\\Program Files\\Tesseract-OCR\\*', 'Tesseract-OCR'),

        ('C:\\Program Files\\ZBar\\bin\\*', 'zbar')

    ],

    hiddenimports=hiddenimports,

    hookspath=[],

    runtime_hooks=[],

    excludes=[],

    win_no_prefer_redirects=False,

    win_private_assemblies=False,

    noarchive=False

)
 
pyz = PYZ(a.pure)
 
exe = EXE(

    pyz,

    a.scripts,

    [],

    exclude_binaries=True,

    name='UnifiedScanner',

    debug=False,

    bootloader_ignore_signals=False,

    strip=False,

    upx=True,

    console=False

)
 
coll = COLLECT(

    exe,

    a.binaries,

    a.zipfiles,

    a.datas,

    strip=False,

    upx=True,

    upx_exclude=[],

    name='UnifiedScanner'

)
 