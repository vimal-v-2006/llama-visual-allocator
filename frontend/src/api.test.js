import { expect, it, vi } from 'vitest';
import { authenticate, api } from './api';
it('exchanges a launch nonce, removes it, and authenticates API requests', async () => {
 history.replaceState(null, '', '/#launch=single-use');
 global.fetch = vi.fn().mockResolvedValueOnce({ok:true,json:async()=>({token:'session-token'})}).mockResolvedValueOnce({ok:true,json:async()=>({model:null})});
 await authenticate(); await api('/state');
 expect(fetch.mock.calls[0][0]).toBe('/api/bootstrap');
 expect(JSON.parse(fetch.mock.calls[0][1].body)).toEqual({nonce:'single-use'});
 expect(location.hash).toBe('');
 expect(fetch.mock.calls[1][1].headers.Authorization).toBe('Bearer session-token');
});
