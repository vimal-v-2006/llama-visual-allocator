export async function authenticate() {
 const nonce = new URLSearchParams(location.hash.slice(1)).get('launch');
 if (nonce) {
  history.replaceState(null, '', location.pathname + location.search);
  const response = await fetch('/api/bootstrap', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({nonce})});
  const data = await response.json();
  if (!response.ok || !data.token) throw new Error(typeof data.detail === 'string' ? data.detail : 'Launch link expired. Open a fresh link from the launcher.');
  sessionStorage.setItem('allocator-token', data.token);
 }
 if (!sessionStorage.getItem('allocator-token')) throw new Error('Authentication required. Open this workspace using the local launcher link.');
}
export async function api(path, body, signal) {
 const response = await fetch('/api' + path, {method:body === undefined?'GET':'POST',headers:{'Content-Type':'application/json',Authorization:'Bearer '+(sessionStorage.getItem('allocator-token') || '')},...(body === undefined?{}:{body:JSON.stringify(body)}),signal});
 const data = await response.json();
 if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `Request failed (${response.status})`);
 return data;
}
