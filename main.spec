# -*- mode: python -*-

import sys
sys.setrecursionlimit(5000)
block_cipher = None


a = Analysis(['main.py'],
             pathex=['./BioenergeticHSC'],
             binaries=[],
             datas=[('./MainUI.ui', '.')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='BioenergeticHSC',
          debug=True,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=False)
app = BUNDLE(exe,
            name='BioenergeticHSC.app',
            icon=None,
            info_plist={
                'NSHighResolutionCapable': 'True'
                },
            )



