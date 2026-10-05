const form=document.getElementById('join-form'),code=document.getElementById('invite-code'),status=document.getElementById('join-status'),button=document.getElementById('join-submit');
const invitation=new URLSearchParams(location.hash.slice(1)).get('invite');
if(invitation){code.value=invitation;history.replaceState(null,'',location.pathname)}
async function join(){button.disabled=true;status.textContent='正在进入实验室…';try{const r=await fetch('/api/join',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({invite:code.value.trim()})});const data=await r.json();if(!r.ok)throw Error(data.error||'无法进入');location.replace('/')}catch(e){status.textContent=e.message;button.disabled=false}}
form.onsubmit=e=>{e.preventDefault();join()};if(invitation)join();
