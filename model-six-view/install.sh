/Applications/Blender.app/Contents/Resources/4.0/python/bin/python3.10 -m ensurepip
/Applications/Blender.app/Contents/Resources/4.0/python/bin/python3.10 -m pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
/Applications/Blender.app/Contents/Resources/4.0/python/bin/python3.10 -m pip install pip-tools -i https://mirrors.aliyun.com/pypi/simple
/Applications/Blender.app/Contents/Resources/4.0/python/bin/pip-compile requirements.txt --output-file requirements-locked.txt  