# Gesture Volume Control Using OpenCV Python

## Requirement
> Pastikan anda sudah membaca `Readme` repostory. Semua libary diintsall menggunakan **CMD Anaconda**

- Install Open CV : `pip install opencv-python`
- Install pycaw : `pip install pycaw`
- Buka link pycaw untuk mengcopy code : https://github.com/AndreMiras/pycaw
- Code yang harus di copy :
```py
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)

volume = cast(interface, POINTER(IAudioEndpointVolume))
volume.GetMute()
volume.GetMasterVolumeLevel()
volume.GetVolumeRange()
volume.SetMasterVolumeLevel(-20.0, None) # -20.0 bisa diganti dengan 0
```