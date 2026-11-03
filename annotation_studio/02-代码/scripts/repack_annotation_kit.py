"""Add portable launchers to an existing handoff without regenerating its data."""
import argparse
import hashlib
import json
import shutil
import stat
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from annotation_launchers import write_launchers, write_archive

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def repack(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    archive = destination.with_suffix('.zip')
    if destination.exists() or archive.exists():
        raise FileExistsError('Choose a new output path; originals are not overwritten')
    with zipfile.ZipFile(source) as z:
        entries = []
        roots = set()
        names = set()
        for info in z.infolist():
            p = PurePosixPath(info.filename)
            if p.is_absolute() or '..' in p.parts or '\\' in info.filename or ':' in info.filename or not p.parts:
                raise ValueError('Unsafe archive path')
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError('Archive symlinks are not supported')
            roots.add(p.parts[0])
            if not info.is_dir():
                relative = Path(*p.parts[1:])
                if len(p.parts) < 2 or relative.as_posix().casefold() in names:
                    raise ValueError('Invalid or duplicate file path')
                names.add(relative.as_posix().casefold())
                entries.append((info, relative))
        if len(roots) != 1:
            raise ValueError('Expected one package root')
        for info, relative in entries:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, target.open('wb') as dst:
                shutil.copyfileobj(src, dst)
    if not (destination/'demo/annotate.html').is_file():
        raise ValueError('Annotation page missing')
    manifest_path = destination/'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    # Validate the incoming manifest before changing any package content.
    for item in manifest['files']:
        p = PurePosixPath(item['path'])
        if p.is_absolute() or '..' in p.parts or '\\' in item['path'] or ':' in item['path']:
            raise ValueError('Unsafe manifest path')
        target = destination.joinpath(*p.parts)
        if target.stat().st_size != item['size'] or sha(target) != item['sha256']:
            raise ValueError('Incoming checksum mismatch: '+item['path'])
    protected = {p.relative_to(destination).as_posix():sha(p) for folder in ['data','demo'] for p in (destination/folder).rglob('*') if p.is_file()}
    write_launchers(destination)
    guide = destination/'使用说明.txt'
    old_guide = guide.read_text(encoding='utf-8-sig')
    guide.write_text('启动兼容性更新：Windows / Mac 推荐用浏览器打开 START_HERE.html。\nMac可双击“启动标注_Mac.command”；权限受限时直接打开HTML。\n详细说明见“启动说明_Windows_Mac.txt”。以下为原包标注说明。\n\n'+old_guide, encoding='utf-8-sig')
    for name in ['annotation_launchers.py','repack_annotation_kit.py']:
        shutil.copy2(Path(__file__).with_name(name), destination/'source'/name)
    manifest.update(package_version=2, launcher_platforms=['Windows','macOS'], repacked_at=datetime.now(timezone.utc).isoformat(), source_archive_sha256=sha(source))
    manifest['files']=[{'path':p.relative_to(destination).as_posix(),'size':p.stat().st_size,'sha256':sha(p)} for p in sorted(destination.rglob('*')) if p.is_file() and p != manifest_path]
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    write_archive(destination, archive)
    with zipfile.ZipFile(archive) as z:
        bad=z.testzip()
        if bad:raise ValueError('ZIP CRC mismatch: '+bad)
        root=destination.name+'/'
        for rel,digest in protected.items():
            if hashlib.sha256(z.read(root+rel)).hexdigest()!=digest:
                raise ValueError('Data or page changed: '+rel)
        info=z.getinfo(root+'启动标注_Mac.command')
        assert info.create_system == 3 and (info.external_attr >> 16) & 0o111
    archive.with_suffix('.zip.sha256.txt').write_text(sha(archive)+'  '+archive.name+'\n',encoding='utf-8')
    print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'protected_files_unchanged':len(protected),'frame_count':manifest.get('frame_count'),'raw_file_count':manifest.get('raw_file_count')},ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    repack(args.input,args.output)
