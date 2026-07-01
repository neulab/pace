"""Download and install VAL (plan validator) without needing unzip."""
import urllib.request
import zipfile
import os
import shutil

val_dir = os.path.expanduser("~/.planutils/packages/val")
os.makedirs(val_dir, exist_ok=True)

url = "https://dev.azure.com/schlumberger/4e6bcb11-cd68-40fe-98a2-e3777bfec0a6/_apis/build/builds/64/artifacts?artifactName=linux64&api-version=6.0&%24format=zip"

print("Downloading VAL...")
zip_path = os.path.join(val_dir, "val.zip")
urllib.request.urlretrieve(url, zip_path)

print("Extracting outer zip...")
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall(val_dir)

inner_zip = os.path.join(val_dir, "linux64", "Val-20210401.1-Linux.zip")
print("Extracting inner zip...")
with zipfile.ZipFile(inner_zip, 'r') as z:
    z.extractall(val_dir)

# Move files from Val-20210401.1-Linux/ to val_dir
src_dir = os.path.join(val_dir, "Val-20210401.1-Linux")
for item in os.listdir(src_dir):
    s = os.path.join(src_dir, item)
    d = os.path.join(val_dir, item)
    if os.path.exists(d):
        if os.path.isdir(d):
            shutil.rmtree(d)
        else:
            os.remove(d)
    shutil.move(s, d)

os.rmdir(src_dir)
os.remove(zip_path)
shutil.rmtree(os.path.join(val_dir, "linux64"), ignore_errors=True)

# Make binaries executable
bin_dir = os.path.join(val_dir, "bin")
if os.path.exists(bin_dir):
    for f in os.listdir(bin_dir):
        os.chmod(os.path.join(bin_dir, f), 0o755)

# Check
validate_path = os.path.join(bin_dir, "Validate") if os.path.exists(bin_dir) else None
if validate_path and os.path.exists(validate_path):
    print(f"VAL installed successfully at {bin_dir}")
    print(f"Set VAL env var: export VAL={bin_dir}")
else:
    print("Files extracted:")
    for root, dirs, files in os.walk(val_dir):
        for f in files:
            print(os.path.join(root, f))
