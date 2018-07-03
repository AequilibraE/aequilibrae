import mock
autodoc_mock_imports = ['sip', 'PyQt5', 'PyQt5.QtGui', 'PyQt5.QtCore', 'PyQt5.QtWidgets']
sys.modules.update((mod_name, mock.MagicMock()) for mod_name in MOCK_MODULES)