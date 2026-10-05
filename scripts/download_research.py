"""Paginated, bounded downloads with rate-limit backoff. No signed URLs are logged."""
import argparse
from pathlib import Path
import re
import time
import requests


def _retry_request(call, *, sleep=time.sleep):
    for attempt in range(5):
        try:
            return call()
        except requests.RequestException as exc:
            code = exc.response.status_code if exc.response is not None else None
            transient = code in (408, 429, 500, 502, 503, 504) or code is None
            if not transient or attempt == 4:
                # Do not expose signed download URLs, authentication, or server bodies.
                raise RuntimeError(f'Request failed: HTTP {code}' if code else
                                   f'Request failed: {type(exc).__name__}') from None
            delay = 45 if code == 429 else min(30, 2 ** (attempt+1))
            print(f'Transient request failure; retrying in {delay}s.', flush=True)
            sleep(delay)


def download(kernel, output, pattern):
    from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest
    api=KaggleApi();api.authenticate()
    owner,slug,version=api.parse_kernel_string(kernel)
    if version is not None:
        raise ValueError('Numeric kernel versions are not API version labels. Use the unversioned kernel slug after checking its current status; archive the returned report and source snapshot together.')
    output.mkdir(parents=True,exist_ok=True)
    matcher=re.compile(pattern) if pattern else None
    token=None;page=0;found=[]
    with api.build_kaggle_client() as client:
        while True:
            request=ApiListKernelSessionOutputRequest()
            request.user_name=owner;request.kernel_slug=slug
            api._set_paging(request,100,token)
            response=_retry_request(lambda: client.kernels.kernels_api_client.list_kernel_session_output(request))
            page+=1
            for item in response.files or []:
                if matcher and not matcher.search(item.file_name):continue
                target=(output/item.file_name).resolve()
                if not target.is_relative_to(output.resolve()):raise ValueError('Unsafe output path')
                if target.exists() and target.stat().st_size>0:
                    found.append(item.file_name);continue
                target.parent.mkdir(parents=True,exist_ok=True)
                temporary=target.with_suffix(target.suffix+'.download')
                with requests.get(item.url,stream=True,timeout=(30,120)) as r:
                    if r.status_code!=200:raise RuntimeError(f'Output download failed: HTTP {r.status_code}')
                    with temporary.open('wb') as f:
                        for block in r.iter_content(1024*1024):f.write(block)
                temporary.replace(target);found.append(item.file_name)
                print('Downloaded',item.file_name,target.stat().st_size,flush=True)
            if page==1 and response.log:(output/(slug+'.log')).write_text(response.log,encoding='utf-8')
            token=response.next_page_token
            if not token:break
            if page%10==0:print('Listed',page*100,'files; matches',len(found),flush=True)
            time.sleep(1)
    if not found:raise RuntimeError('No matching outputs; wait until the notebook is complete.')
    print('Complete:',len(found),'matched files.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kernel',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--pattern')
    a=p.parse_args();download(a.kernel,a.output,a.pattern)
