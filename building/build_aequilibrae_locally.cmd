rem  THIS batch runs all the wheel builds & the AoN builds for QGIS for Linux on Windows 32 & 64 bits

docker run -v F:/SourceCode/aequilibrae://tmp/mr -w //tmp/mr python35 python3 setup.py sdist bdist_wheel
docker run -v F:/SourceCode/aequilibrae://tmp/mr -w //tmp/mr python36 python3 setup.py sdist bdist_wheel
docker run -v F:/SourceCode/aequilibrae://tmp/mr -w //tmp/mr python37 python3 setup.py sdist bdist_wheel
rmdir ..\aequilibrae\aequilibrae.egg-info /S/Q

docker run -v F:/SourceCode/aequilibrae/aequilibrae/paths://tmp/mr -w //tmp/mr python35 python3 setup_Assignment.py build_ext --inplace
docker run -v F:/SourceCode/aequilibrae/aequilibrae/paths://tmp/mr -w //tmp/mr python36 python3 setup_Assignment.py build_ext --inplace
docker run -v F:/SourceCode/aequilibrae/aequilibrae/paths://tmp/mr -w //tmp/mr python37 python3 setup_Assignment.py build_ext --inplace
rmdir ..\aequilibrae\aequilibrae.egg-info /S/Q


c:\python35\python.exe setup.py sdist bdist_wheel
c:\python36\python.exe setup.py sdist bdist_wheel
c:\python37\python.exe setup.py sdist bdist_wheel
rmdir ..\aequilibrae\aequilibrae.egg-info /S/Q

c:\Python_32Bits\Python35-32\python.exe setup.py sdist bdist_wheel
c:\Python_32Bits\Python36-32\python.exe setup.py sdist bdist_wheel
c:\Python_32Bits\Python37-32\python.exe setup.py sdist bdist_wheel
rmdir ..\aequilibrae\aequilibrae.egg-info /S/Q

cd ..\aequilibrae\paths
c:\python35\python.exe setup_Assignment.py build_ext --inplace
c:\python36\python.exe setup_Assignment.py build_ext --inplace
c:\python37\python.exe setup_Assignment.py build_ext --inplace
c:\Python_32Bits\Python35-32\python.exe setup_Assignment.py build_ext --inplace
c:\Python_32Bits\Python36-32\python.exe setup_Assignment.py build_ext --inplace
c:\Python_32Bits\Python37-32\python.exe setup_Assignment.py build_ext --inplace