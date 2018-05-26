## easy-eye-tracker
This is an OpenCV based webcam gaze tracker based on [trishume project](https://github.com/trishume/eyeLike).


## Status
The eye center tracking works well but the gaze tracker is currently on development.

## Building

CMake is required to build eyeLike.

Also you need OpenCV and X11 (the last one just for Linux).

### Linux
You can build your own version of OpenCV or you can just download it from official repos.


**OpenCV:**
```bash
sudo apt install opencv
```
**X11:**
```bash
sudo apt install libx11-dev
```

**Build**
```bash
# do things in the build directory so that we don't clog up the main directory
mkdir build
cd build
cmake ../
make
./bin/eyeLike # the executable file
```

### OSX
We don't own a mac so we can't really say how to build on OSX

### Windows
We are currently working on it
