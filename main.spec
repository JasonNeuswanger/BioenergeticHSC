# -*- mode: python -*-
import sys
sys.setrecursionlimit(5000)
block_cipher = None


if sys.platform == 'darwin':
    a = Analysis(['main.py'],
             pathex=['./BioenergeticHSC'],
             binaries=[],
             datas=[('./MainUi.ui', '.')],
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
          console=False)
    app = BUNDLE(exe,
            name='BioenergeticHSC.app',
            icon=None,
            info_plist={
                'NSHighResolutionCapable': 'True'
                },
            )
if sys.platform == 'win32' or sys.platform == 'win64' or sys.platform == 'linux':
    a = Analysis(['main.py'],
             pathex=['.\BioenergeticHSC'],
             datas=[('.\MainUi.ui', '.')],
             binaries=[],
             hiddenimports=['pkg_resources.py2_warn'],
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
          [('W ignore', None, 'OPTION')],
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='BioenergeticHSC',
          debug=True,
          bootloader_ignore_signals=False,
          strip=False,
          upx=False,
          console=True)

