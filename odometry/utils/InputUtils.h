#pragma once

// Win32 keyboard polling shared by the AirSim entrypoints (FreeplayDrone,
// PathPlayer, PathRecorder). Windows-only; every target that uses it is gated
// on WIN32 in CMakeLists, so the header is empty on other platforms.
#ifdef _WIN32
#include <windows.h>

inline bool isKeyPressed(int key) {
    return (GetAsyncKeyState(key) & 0x8000) != 0;
}
#endif
