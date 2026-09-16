"""Fail-closed publication checks for Git history, index, and allowed files.

Reports paths and finding categories only. Never print matched secret values.
This complements human data-classification review; it is not a secrecy proof.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DENIED_SUFFIXES = {'.xls', '.xlsx', '.xlsm', '.xlsb', '.csv', '.tsv', '.json', '.jsonl',
                   '.ndjson', '.db', '.sqlite', '.sqlite3', '.joblib', '.pkl', '.pickle',
                   '.pt', '.pth', '.cbm', '.onnx', '.h5', '.hdf5', '.npy', '.npz',
                   '.doc', '.docx', '.pdf', '.ppt', '.pptx', '.zip', '.7z', '.rar',
                   '.pem', '.key', '.p12', '.pfx', '.log', '.ipynb'}
DENIED_ROOTS = ('data/', 'revise/', 'srFigure/', '第一次返修前/', '.runtime/', '.local-only/',
                '.codex/', '.agents/', '.claude/', '.venv/', 'node_modules/')
REVIEWED_IMAGES = {'docs/images/demo-overview.png', 'docs/images/demo-optimization.png'}
PATTERNS = {
    'github_token': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b'),
    'private_key': re.compile(r'-----BEGIN (?:[A-Z ]+)?PRIVATE KEY-----'),
    'aws_access_key': re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'openai_key': re.compile(r'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}\b'),
    'literal_credential': re.compile(r'''(?i)(?:password|passwd|api[_-]?key|client[_-]?secret|access[_-]?token)\s*[:=]\s*["'][^"'\s]{8,}["']'''),
    'credential_url': re.compile(r'(?i)\b[a-z][a-z0-9+.-]*://[A-Za-z0-9._~%+-]+:[A-Za-z0-9._~%+-]{3,}@'),
}


def git(*args):
    return subprocess.check_output(['git', '-c', 'core.quotepath=false',
                                    '-c', 'core.excludesFile='+os.devnull, *args], cwd=ROOT)


def audit_blob(path, content, source):
    findings = []
    normalized = path.replace('\\', '/')
    suffix = Path(normalized).suffix.lower()
    name = Path(normalized).name.lower()
    if normalized.casefold().startswith(tuple(p.casefold() for p in DENIED_ROOTS)) or suffix in DENIED_SUFFIXES or name.startswith('.env'):
        findings.append({'path':path, 'source':source, 'type':'private_or_data_file'})
    if normalized in REVIEWED_IMAGES:
        if not content.startswith(b'\x89PNG\r\n\x1a\n'):
            findings.append({'path':path, 'source':source, 'type':'unexpected_image_format'})
        return findings
    try:
        text = content.decode('utf-8-sig')
    except UnicodeDecodeError:
        return findings + [{'path':path, 'source':source, 'type':'unreviewed_binary'}]
    if len(content) > 1_000_000:
        findings.append({'path':path, 'source':source, 'type':'unexpected_large_text'})
    for category, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            findings.append({'path':path, 'source':source, 'type':category,
                             'line':text.count('\n', 0, match.start())+1})
    return findings


def run(staged_only=False):
    findings=[]; checked=set(); files={}; content_cache={}; inspected_paths=set()
    for item in git('ls-files','--stage','-z').split(b'\0'):
        if not item: continue
        spec,name=item.split(b'\t',1); mode,oid,stage=spec.decode().split()
        path=name.decode('utf-8');files[path]=oid
        ignored = subprocess.run(['git', '-c', 'core.excludesFile='+os.devnull,
                                  'check-ignore', '--quiet', '--no-index', path], cwd=ROOT)
        if ignored.returncode == 0:
            findings.append({'path':path,'source':'index','type':'not_in_public_allowlist'})
        if mode not in ('100644','100755') or stage!='0':
            findings.append({'path':path,'source':'index','type':'unsupported_mode_or_conflict'})
        content=content_cache.setdefault(oid, git('cat-file','blob',oid))
        findings.extend(audit_blob(path,content,'index'))
        checked.add(oid)
        inspected_paths.add((oid,path))
    if not staged_only:
        commits=git('rev-list','--all').decode().splitlines()
        for commit in commits:
            findings.extend(audit_blob('commit:'+commit,git('cat-file','commit',commit),'commit_metadata'))
            for item in git('ls-tree','-r','-z',commit).split(b'\0'):
                if not item: continue
                spec,name=item.split(b'\t',1)
                mode,kind,oid=spec.decode().split(); path=name.decode('utf-8')
                if (oid,path) in inspected_paths: continue
                inspected_paths.add((oid,path))
                if kind!='blob' or mode not in ('100644','100755'):
                    findings.append({'path':path,'source':'history','type':'unreviewed_gitlink_or_mode'})
                    continue
                if oid not in content_cache:content_cache[oid]=git('cat-file','blob',oid)
                findings.extend(audit_blob(path,content_cache[oid],'history'))
                checked.add(oid)
        for entry in git('for-each-ref','--format=%(objectname) %(objecttype)','refs/tags').decode().splitlines():
            oid,kind=entry.split()
            if kind=='tag':
                findings.extend(audit_blob('tag:'+oid,git('cat-file','tag',oid),'tag_metadata'))
    manifest='\n'.join(path+' '+oid for path,oid in sorted(files.items()))
    return {'scope':'index' if staged_only else 'all reachable refs and index',
            'indexed_files':len(files),'checked_blobs':len(checked),
            'manifest_sha256':hashlib.sha256(manifest.encode()).hexdigest(),
            'findings':findings,'passed':not findings}


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staged-only',action='store_true')
    args=parser.parse_args()
    report=run(args.staged_only)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    sys.exit(0 if report['passed'] else 1)
