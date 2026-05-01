import json
import streamlit.components.v1 as components

def login_state_extender(email):
    # email を JS に埋め込む（必ず JSON エスケープ）
    email_js = json.dumps(email or "")

    html = """<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <title>EasyAuth Keepalive (minimal + BC + jitter)</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    #auth { font-family: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; margin: 10px }
    #login { display: none; margin-top: 8px }
    .btn { display:inline-block; padding:6px 10px; background:#0b69ff; color:#fff; text-decoration:none; border-radius:4px }
    #status { color:#666; font-size:.9em; margin-top:6px }
  </style>
</head>
<body>
  <div id="auth">
    <div id="info">認証状態を確認しています</div>
    <div id="login">
      <a id="loginLink" class="btn" target="_blank" rel="noopener">ログイン</a>
      <div id="status"></div>
    </div>
  </div>

  <script>
  (function(){
    'use strict';

    const SAVED_EMAIL = """ + f'{email_js}' + """;

    // 調整ポイント
    const LOGIN_URL        = '/.auth/login/google?post_login_redirect_uri=/&access_type=offline' + (SAVED_EMAIL ? '&login_hint=' + encodeURIComponent(SAVED_EMAIL) : '');
    const AUTH_ME_URL      = '/.auth/me';
    const REFRESH_URL      = '/.auth/refresh';

    const REFRESH_EARLY_MS = 5*60*1000;   // 期限の5分前に更新
    const NORMAL_POLL_MS   = 60*1000;     // 未認証時の定期確認
    const LOGIN_POLL_MS    = 3000;        // ログイン後の短期確認
    const LOGIN_POLL_MAX   = 90*1000;     // 短期確認の最大時間

    const HEARTBEAT_MS     = 120*1000;    // 任意: 心拍（/.auth/me）。不要なら 0 に
    const JITTER_MS        = 30000;       // タブ固定ジッター幅（0〜30秒）
    const TAB_JITTER       = Math.floor(Math.random() * JITTER_MS);

    const info   = document.getElementById('info');
    const loginW = document.getElementById('login');
    const link   = document.getElementById('loginLink');
    const status = document.getElementById('status');
    link.href = LOGIN_URL;

    let bc = null; try { bc = new BroadcastChannel('auth-login'); } catch(e) {}

    let refreshTimer = null;
    let normalPollTimer = null;
    let shortPollTimer = null;
    let heartbeatTimer = null;
    let scheduledExpiry = null; // 目標の有効期限(ms)を保存

    function showLogin(msg) {
      if (msg) info.textContent = msg;
      console.log('[keepalive] showLogin', msg);
      status.textContent = '';
      loginW.style.display = 'block';
    }

    function hideLogin(msg) {
      if (msg) info.textContent = msg;
      console.log('[keepalive] hideLogin', msg);
      status.textContent = '';
      loginW.style.display = 'none';
    }

    async function getMe() {
      const r = await fetch(AUTH_ME_URL, { credentials:'include', cache:'no-store' }).catch(err => {
        console.error('[keepalive] getMe failed', err);
        return null;
      });
      if (!r) return null;
      console.log('[keepalive] getMe', r.status);
      if (!r.ok) return null;
      try {
        const json = await r.json();
        console.log('[keepalive] getMe', json);
        return json;
      } catch {
        return null;
      }
    }
    // App Service EasyAuth は配列で返ることが多い
    function pickEntry(me) {
      if (!me) return null;
      if (Array.isArray(me)) return me[0] || null;
      if (me.clientPrincipal) return me.clientPrincipal; // 互換
      return me;
    }
    function getExpiryMs(me) {
      const e = pickEntry(me);
      if (!e) return null;
      if (e.expires_on) return new Date(e.expires_on).getTime();
      if (e.exp) return (typeof e.exp === 'number' ? e.exp*1000 : new Date(e.exp).getTime());
      return null;
    }
    function isAuthed(me) {
      const e = pickEntry(me);
      return !!(e && (e.expires_on || e.user_id || e.userId || e.id_token || e.access_token));
    }

    function scheduleNextRefresh(expiryMs) {
      if (!expiryMs) { startNormalPoll(); return; }
      // 同一期限の重複スケジュール抑止（±1秒以内で既にタイマーありならスキップ）
      if (scheduledExpiry && Math.abs(expiryMs - scheduledExpiry) < 1000 && refreshTimer) return;
      scheduledExpiry = expiryMs;
      if (refreshTimer) { clearTimeout(refreshTimer); refreshTimer = null; }
      const base = Math.max(expiryMs - Date.now() - REFRESH_EARLY_MS, 3000);
      const delay = Math.max(3000, base + TAB_JITTER);
      console.log('[keepalive] schedule next refresh ' + new Date(Date.now() + delay).toLocaleString('ja-JP', { timeZone: 'Asia/Tokyo', hour12: false }) + ' JST');
      refreshTimer = setTimeout(checkThenRefresh, delay);
    }

    async function checkThenRefresh() {
      const me = await getMe();
      if (!isAuthed(me)) { showLogin('再ログインが必要です'); startNormalPoll(); return; }
      const exp = getExpiryMs(me);
      // 他タブが先に更新して期限が伸びていたら更新せず再スケジュール
      if (exp && (exp - Date.now() > REFRESH_EARLY_MS)) { scheduleNextRefresh(exp); return; }
      await refreshOnce();
    }

    async function refreshOnce() {
      try {
        console.log('[keepalive] refresh start');
        const r = await fetch(REFRESH_URL, { method:'POST', credentials:'include', cache:'no-store' });
        if (!r.ok) {
          console.log('[keepalive] refresh failed', r.status);
          if (r.status === 403) { showLogin('再ログインが必要です'); startNormalPoll(); return; }
          startNormalPoll(); return;
        }
      } catch { startNormalPoll(); return; }
      console.log('[keepalive] refresh finished');

      const me = await getMe();
      const exp = getExpiryMs(me);
      if (isAuthed(me) && exp && (exp - Date.now() > REFRESH_EARLY_MS)) {
        hideLogin('認証済みです');
        scheduleNextRefresh(exp);
        if (bc) try { bc.postMessage({ type:'authed', exp }); } catch(e) {}
      } else {
        showLogin('再ログインが必要です');
        startNormalPoll();
      }
    }

    async function startNormalPoll() {
      if (normalPollTimer) return;
      const me0 = await getMe();
      const exp0 = getExpiryMs(me0);
      let lastRun = 0;
      normalPollTimer = setInterval(async () => {
        const now = Date.now();
        if (now - lastRun < NORMAL_POLL_MS) return;
        lastRun = now;
        const me = await getMe();
        const exp = getExpiryMs(me);
        if (isAuthed(me) && exp && exp != exp0 && (exp - Date.now() > REFRESH_EARLY_MS)) {
          clearInterval(normalPollTimer); normalPollTimer = null;
          hideLogin('認証済みです');
          scheduleNextRefresh(exp);
          if (bc) try { bc.postMessage({ type:'authed', exp }); } catch(e) {}
        }
      }, NORMAL_POLL_MS);
    }

    async function startShortPoll(waitMsg, timeoutMsg) {
      if (shortPollTimer) { clearInterval(shortPollTimer); shortPollTimer = null; }
      const start = Date.now();
      const me0 = await getMe();
      const exp0 = getExpiryMs(me0);
      let lastRun = 0;
      if (waitMsg) { info.textContent = waitMsg; status.textContent = ''; }
      shortPollTimer = setInterval(async () => {
        const now = Date.now();
        if (now - lastRun < LOGIN_POLL_MS) return;
        lastRun = now;
        if (Date.now() - start > LOGIN_POLL_MAX) {
          clearInterval(shortPollTimer); shortPollTimer = null;
          showLogin(timeoutMsg || 'ログインが完了しませんでした。もう一度お試しください');
          startNormalPoll();
          return;
        }
        const me = await getMe();
        const exp = getExpiryMs(me);
        if (isAuthed(me) && exp && exp != exp0 && (exp - Date.now() > REFRESH_EARLY_MS)) {
          clearInterval(shortPollTimer); shortPollTimer = null;
          hideLogin('ログインが完了しました');
          scheduleNextRefresh(exp);
          if (bc) try { bc.postMessage({ type:'authed', exp }); } catch(e) {}
        }
      }, LOGIN_POLL_MS);
    }

    // SAVED_EMAIL が空ならここで完全に停止
    if (!SAVED_EMAIL) {{
      hideLogin('ローカルモード: keepalive停止');
      console.log('[keepalive] disabled. email is empty (e.g. local mode).');
      return;
    }}

    link.addEventListener('click', () => {
      hideLogin('ログインを開始しました。完了を待ちます');
      if (bc) try { bc.postMessage({ type:'login-clicked' }); } catch(e) {}
      startShortPoll(null, 'ログインが完了しませんでした。もう一度お試しください');
    });

    let lastAuthed = 0;
    if (bc) {
      bc.onmessage = (ev) => {
        const m = ev.data;
        if (!m || !m.type) return;
        if (m.type === 'login-clicked') {
          console.log('[keepalive] login-clicked');
          hideLogin('別タブでログインが開始されました。完了を待ちます');
          startShortPoll(null, '別タブでのログインがタイムアウトしました');
        } else if (m.type === 'authed') {
          lastAuthed = Date.now();
          console.log('[keepalive] authed');
          hideLogin('認証済みです');
          if (m.exp) scheduleNextRefresh(m.exp);
        }
      };
    }

    if (HEARTBEAT_MS > 0) {
      let lastRun = 0;
      heartbeatTimer = setInterval(async () => {
        const now = Date.now();
        if (now - lastRun < HEARTBEAT_MS || now - lastAuthed < HEARTBEAT_MS) return;
        lastRun = now;
        if (normalPollTimer || shortPollTimer) return;
        console.log('[keepalive] heartbeat');
        const me = await getMe();
        const exp = getExpiryMs(me);
        if (!(isAuthed(me) && exp && (exp - Date.now() > REFRESH_EARLY_MS))) {
          showLogin('再ログインが必要です');
          startNormalPoll();
        }
      }, HEARTBEAT_MS);
    }

    // 初期化: 問答無用の refresh → me で判定
    (async function init(){
      await refreshOnce();
    })();

    window.addEventListener('beforeunload', () => {
      if (bc) try { bc.close(); } catch(e) {}
      if (refreshTimer) clearTimeout(refreshTimer);
      if (normalPollTimer) clearInterval(normalPollTimer);
      if (shortPollTimer) clearInterval(shortPollTimer);
      if (heartbeatTimer) clearInterval(heartbeatTimer);
    });
  })();
  </script>
</body>
</html>
"""
    components.html(html, height=80)
