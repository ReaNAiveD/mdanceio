import os
import re

if os.getenv("GITHUB_REF_TYPE") != "tag":
    print("Action is not Triggered by TAG")
    exit(1)

version = os.getenv("GITHUB_REF_NAME")

if not isinstance(version, str):
    print("Version tag is not str")
    exit(1)

if not version.startswith("v"):
    print("Version tag is not started with 'v'")
    exit(1)

version_num = version[1:]

if not re.match(r'[0-9.]+(-SNAPSHOT)?', version_num):
    print("Version format is incorrect")

with open('build.gradle', mode='r', encoding='utf-8') as gradle_file:
    version_in_gradle = ""
    in_ext_library = False
    for line in gradle_file.readlines():
        if "ext.library" in line:
            in_ext_library = True
        if in_ext_library:
            version_match = re.match(r'\s*version\s*:\s*[\'\"]([0-9.]+(-SNAPSHOT)?)[\'\"],?\s*', line)
            if version_match:
                version_in_gradle = version_match.group(1)
                break
            if "]" in line:
                break
    if not version_in_gradle:
        print("Version number not found in build.gradle")
        exit(1)
    if version_in_gradle != version_num:
        print("Version number in Gradle mismatch the TAG")
        exit(1)

with open('mdanceio/Cargo.toml', mode='r', encoding='utf-8') as mdanceio_toml:
    version_in_mdanceio = ""
    in_package = False
    for line in mdanceio_toml.readlines():
        if "[package]" in line:
            in_package = True
        if in_package:
            version_match = re.match(r'version\s*=\s*\"([0-9.]+)\"\s*', line)
            if version_match:
                version_in_mdanceio = version_match.group(1)
                break
    if not version_in_mdanceio:
        print("Version number not found in mdanceio/Cargo.toml")
        exit(1)
    if version_in_mdanceio != version_num:
        print("Version number in mdanceio mismatch the TAG")
        exit(1)

with open('nanoem/Cargo.toml', mode='r', encoding='utf-8') as nanoem_toml:
    version_in_nanoem = ""
    in_package = False
    for line in nanoem_toml.readlines():
        if "[package]" in line:
            in_package = True
        if in_package:
            version_match = re.match(r'version\s*=\s*\"([0-9.]+)\"\s*', line)
            if version_match:
                version_in_nanoem = version_match.group(1)
                break
    if not version_in_nanoem:
        print("Version number not found in nanoem/Cargo.toml")
        exit(1)
    if version_in_nanoem != version_num:
        print("Version number in nanoem mismatch the TAG")
        exit(1)
