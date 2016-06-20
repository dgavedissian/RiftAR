RiftAR
========================

Note
----------------
Objects that are expected to have a single owner, such as a camera or entity, are stored in a
unique_ptr. Resources on the other hand such as Models and Shaders which can be shared between many
entities are stored in a shared_ptr.

This project depends on:
    - Oculus SDK 1.3
    - Latest ZED SDK
    - CUDA 7.5