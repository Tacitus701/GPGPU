# GPGPU

# Compilation
```
mkdir build
cd build
cmake ..
make <target>
```
where \<target\> can be :
- cpu
- gpu

# GPU Usage
```
mkdir res
./gpu <filename>
```
<b>Warning : you need to have a folder called <u>res</u> for the output of the executable !</b>

# Find barcode in image at filename, result is in barcode.png

./cpu filename
