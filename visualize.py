"""Pygame visualization of CartPole balanced by the connectome readout.

Two windows: the cart-pole sim, and a grid of all 50K subcircuit neurons
(lit when spiking; cyan=sensory, orange/purple=motor pools).

The spike window runs in its own process (pygame can't open a second
window via set_mode — it just resizes the first).

Usage:
  uv run python visualize.py
  uv run python visualize.py --fps 30 --episodes 5
  uv run python visualize.py --no-truncate   # ignore 500-step cap, run until it falls
  uv run python visualize.py --no-spikes     # cart-pole window only
"""
import argparse
import multiprocessing as mp
import pickle
from multiprocessing import shared_memory

import gymnasium as gym
import numpy as np
import pygame
import scipy.sparse as sp

from neuron import LIFNeuron
from sensors import encode_cartpole_state

# Colors
BG = (18, 18, 24)
TRACK = (60, 60, 70)
CART = (80, 180, 255)
POLE = (255, 120, 80)
TEXT = (220, 220, 220)
ACCENT = (120, 255, 160)
BAD = (255, 90, 90)
PANEL = (28, 28, 36)
FIRE = np.array([120, 255, 160], dtype=np.uint8)
SENS_COLOR = np.array([80, 200, 255], dtype=np.uint8)
MPOS_COLOR = np.array([255, 150, 80], dtype=np.uint8)
MNEG_COLOR = np.array([200, 120, 255], dtype=np.uint8)


def load_subcircuit():
    z = np.load('subcircuit_A.npz')
    A = sp.csr_matrix(
        (z['data'], z['indices'], z['indptr']),
        shape=tuple(z['shape']),
    )
    sensory_map = np.load('subcircuit_sensory.npy')
    cfg = np.load('subcircuit_cfg.npy')
    return A, sensory_map, float(cfg[0]), float(cfg[1])


def load_readout(path='readout.pkl'):
    with open(path, 'rb') as f:
        bundle = pickle.load(f)
    if bundle.get('clf_both') is not None:
        return bundle['clf_both'], bundle['feat_both']
    if bundle.get('clf_sens') is not None:
        return bundle['clf_sens'], bundle['feat_sens']
    return bundle['model'], bundle['features']


def encode_sub(state, sensory_map, n, sc):
    s = np.zeros(n, dtype=np.float32)
    s[sensory_map] = encode_cartpole_state(state, 1000)
    return s * sc


def _load_pool(path):
    try:
        return np.load(path)
    except OSError:
        return np.empty(0, dtype=np.int64)


def handle_events():
    for e in pygame.event.get():
        if e.type in (pygame.QUIT, pygame.WINDOWCLOSE):
            return False
        if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
            return False
    return True


def _spike_window_main(spikes_name, ctrl_name, n, sensory, m_pos, m_neg):
    """Child process: owns the second pygame window."""
    shm = shared_memory.SharedMemory(name=spikes_name)
    ctrl = shared_memory.SharedMemory(name=ctrl_name)
    spikes = np.ndarray((n,), dtype=np.uint8, buffer=shm.buf)
    ctrl_a = np.ndarray((4,), dtype=np.int64, buffer=ctrl.buf)

    sensory = np.asarray(sensory, dtype=np.int64)
    m_pos = np.asarray(m_pos, dtype=np.int64)
    m_neg = np.asarray(m_neg, dtype=np.int64)

    cols = 250
    rows = (n + cols - 1) // cols
    grid_n = rows * cols
    scale = 3
    header_h = 64

    pygame.init()
    screen = pygame.display.set_mode((cols * scale, rows * scale + header_h))
    pygame.display.set_caption('Subcircuit spikes (50K neurons)')
    font = pygame.font.SysFont('consolas', 16)
    font_legend = pygame.font.SysFont('consolas', 13)
    small = pygame.Surface((cols, rows))
    px = np.empty((rows, cols, 3), dtype=np.uint8)
    bg = np.array(BG, dtype=np.uint8)
    px[:] = bg

    cls = np.zeros(grid_n, dtype=np.uint8)
    cls[sensory] = 1
    if m_pos.size:
        cls[m_pos] = 2
    if m_neg.size:
        cls[m_neg] = 3
    cls = cls[:grid_n].reshape(rows, cols)
    palette = np.stack([FIRE, SENS_COLOR, MPOS_COLOR, MNEG_COLOR])

    clock = pygame.time.Clock()
    running = True
    while running and ctrl_a[1] == 0:
        for e in pygame.event.get():
            if e.type in (pygame.QUIT, pygame.WINDOWCLOSE):
                running = False
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                running = False
        if not running:
            break

        step = int(ctrl_a[0])
        spk = spikes.astype(bool)
        if spk.size < grid_n:
            spk = np.concatenate(
                [spk, np.zeros(grid_n - spk.size, dtype=bool)],
            )
        spk = spk.reshape(rows, cols)

        px[:] = bg
        px[spk] = palette[cls[spk]]

        # blit_array wants (width, height, 3)
        pygame.surfarray.blit_array(small, px.transpose(1, 0, 2))
        w, h = cols * scale, rows * scale
        screen.blit(pygame.transform.scale(small, (w, h)), (0, header_h))

        screen.fill(BG, (0, 0, w, header_h))
        fired = int(spk.sum())
        pct = 100.0 * fired / n
        hdr = font.render(
            f'step {step}  |  firing {fired} / {n} ({pct:.2f}%)',
            True, TEXT,
        )
        screen.blit(hdr, (8, 6))

        x = 8
        for col, name in (
            (FIRE, 'other'), (SENS_COLOR, 'sensory'),
            (MPOS_COLOR, 'motor+'), (MNEG_COLOR, 'motor-'),
        ):
            pygame.draw.rect(screen, col, (x, 34, 12, 12), border_radius=2)
            s = font_legend.render(name, True, TEXT)
            screen.blit(s, (x + 16, 33))
            x += 24 + s.get_width() + 14

        pygame.display.update()
        clock.tick(30)

    ctrl_a[2] = 1  # child exiting
    shm.close()
    ctrl.close()
    pygame.quit()


class SpikeMapClient:
    """Parent side: shared-memory feed to the spike-window process."""

    def __init__(self, n, sensory_map):
        self.n = n
        self._shm = shared_memory.SharedMemory(
            create=True, size=max(n, 1),
        )
        self._ctrl = shared_memory.SharedMemory(create=True, size=8 * 4)
        self.spikes = np.ndarray((n,), dtype=np.uint8, buffer=self._shm.buf)
        self.ctrl = np.ndarray((4,), dtype=np.int64, buffer=self._ctrl.buf)
        self.ctrl[:] = 0

        m_pos = _load_pool('subcircuit_motor_pos.npy')
        m_neg = _load_pool('subcircuit_motor_neg.npy')
        ctx = mp.get_context('spawn')
        self.proc = ctx.Process(
            target=_spike_window_main,
            args=(
                self._shm.name, self._ctrl.name, n,
                sensory_map.astype(np.int64).tolist(),
                m_pos.astype(np.int64).tolist(),
                m_neg.astype(np.int64).tolist(),
            ),
            daemon=True,
        )
        self.proc.start()

    def update(self, spikes, step):
        if self.proc.is_alive():
            self.spikes[:] = spikes.astype(np.uint8, copy=False)
            self.ctrl[0] = step

    def close(self):
        if self.proc.is_alive():
            self.ctrl[1] = 1
            self.proc.join(timeout=3)
            if self.proc.is_alive():
                self.proc.terminate()
                self.proc.join(timeout=1)
        for shm in (self._shm, self._ctrl):
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass


class Renderer:
    def __init__(self, width=900, height=520):
        pygame.init()
        pygame.display.set_caption('Fly Connectome — CartPole')
        self.screen = pygame.display.set_mode((width, height))
        self.width = width
        self.height = height
        self.font = pygame.font.SysFont('consolas', 18)
        self.font_big = pygame.font.SysFont('consolas', 28, bold=True)
        self.track_y = int(height * 0.72)
        self.scale = (width * 0.85) / 4.8
        self.cx = width // 2
        self.cart_w = 70
        self.cart_h = 36
        self.pole_len = 170

    def world_to_px(self, x):
        return self.cx + x * self.scale

    def draw(self, state, info):
        self.screen.fill(BG)

        pygame.draw.line(
            self.screen, TRACK,
            (40, self.track_y), (self.width - 40, self.track_y), 4,
        )
        for x in np.linspace(-2.4, 2.4, 13):
            px = self.world_to_px(x)
            pygame.draw.line(
                self.screen, TRACK,
                (px, self.track_y - 6), (px, self.track_y + 6), 2,
            )

        cart_x, _, theta, _ = state
        cx_px = self.world_to_px(cart_x)
        cx_px = max(50, min(self.width - 50, cx_px))
        cart_rect = pygame.Rect(
            int(cx_px - self.cart_w / 2),
            self.track_y - self.cart_h,
            self.cart_w, self.cart_h,
        )
        pygame.draw.rect(self.screen, CART, cart_rect, border_radius=6)
        wr = 9
        for off in (-self.cart_w // 3, self.cart_w // 3):
            pygame.draw.circle(
                self.screen, (40, 40, 50),
                (int(cx_px + off), self.track_y + wr + 2), wr,
            )

        tip_x = cx_px + self.pole_len * np.sin(theta)
        tip_y = (self.track_y - self.cart_h) - self.pole_len * np.cos(theta)
        pygame.draw.line(
            self.screen, POLE,
            (int(cx_px), self.track_y - self.cart_h),
            (int(tip_x), int(tip_y)), 8,
        )
        pygame.draw.circle(
            self.screen, (255, 200, 160),
            (int(tip_x), int(tip_y)), 8,
        )

        panel = pygame.Rect(12, 12, 320, 130)
        pygame.draw.rect(self.screen, PANEL, panel, border_radius=8)

        theta_deg = np.degrees(theta)
        color = ACCENT if abs(theta_deg) < 12 else BAD
        lines = [
            f'step   {info["step"]:>5}',
            f'reward {info["reward"]:>5.0f}',
            f'theta  {theta_deg:+7.2f} deg',
            f'x      {cart_x:+7.3f}',
            f'action {"RIGHT >" if info["action"] == 1 else "< LEFT "}',
            f'firing {info["spikes"]:>5} / {info["n"]}',
        ]
        for i, line in enumerate(lines):
            surf = self.font.render(line, True, color if i in (0, 2) else TEXT)
            self.screen.blit(surf, (panel.x + 14, panel.y + 10 + i * 20))

        banner = self.font_big.render(
            f'EP {info["ep"] + 1}/{info["episodes"]}', True, TEXT,
        )
        self.screen.blit(banner, (self.width - banner.get_width() - 16, 16))

        if info.get('msg'):
            msg = self.font_big.render(
                info['msg'], True, info.get('msg_color', ACCENT),
            )
            self.screen.blit(
                msg, ((self.width - msg.get_width()) // 2, 30),
            )

        gauge_w, gauge_h = 200, 14
        gx, gy = self.width - gauge_w - 16, 60
        pygame.draw.rect(
            self.screen, TRACK, (gx, gy, gauge_w, gauge_h), border_radius=7,
        )
        frac = np.clip((theta_deg + 45) / 90.0, 0, 1)
        pygame.draw.rect(
            self.screen, color,
            (gx, gy, int(gauge_w * frac), gauge_h), border_radius=7,
        )
        pygame.draw.line(
            self.screen, TEXT,
            (gx + gauge_w // 2, gy - 3),
            (gx + gauge_w // 2, gy + gauge_h + 3), 2,
        )

    def flash(self, color, ms=350):
        self.screen.fill(color, special_flags=pygame.BLEND_RGB_ADD)
        pygame.display.update()
        pygame.time.delay(ms)


def main():
    parser = argparse.ArgumentParser(description='Visualize connectome CartPole')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--readout', default='readout.pkl')
    parser.add_argument(
        '--no-truncate', action='store_true',
        help='ignore CartPole 500-step cap; episode ends only when pole falls',
    )
    parser.add_argument(
        '--no-spikes', action='store_true',
        help='hide the neuron-spike window',
    )
    args = parser.parse_args()

    print('Loading subcircuit...', flush=True)
    A, sensory_map, thr, sc = load_subcircuit()
    n = A.shape[0]
    print(f'  A {A.shape} nnz={A.nnz} thr={thr} sc={sc}', flush=True)

    print(f'Loading readout ({args.readout})...', flush=True)
    clf, feats = load_readout(args.readout)
    print(f'  features: {feats.size} neurons', flush=True)

    env = gym.make('CartPole-v1')
    r = Renderer()
    smap = None if args.no_spikes else SpikeMapClient(n, sensory_map)
    clock = pygame.time.Clock()
    running = True

    ep = 0
    state, _ = env.reset(seed=args.seed)
    neuron = LIFNeuron(n, decay=0.95, threshold=thr)
    step = 0
    total = 0.0
    msg = ''
    msg_color = ACCENT
    msg_until = 0

    def render(info, spikes):
        nonlocal running
        if not handle_events():
            running = False
            return
        r.draw(state, info)
        pygame.display.update()
        if smap is not None:
            smap.update(spikes, step)
        clock.tick(args.fps)

    try:
        while running and ep < args.episodes:
            spikes = neuron.step(A, encode_sub(state, sensory_map, n, sc))
            action = int(clf.predict(spikes[feats].reshape(1, -1))[0])
            state, reward, terminated, truncated, _ = env.step(action)
            step += 1
            total += reward

            info = {
                'step': step,
                'reward': total,
                'action': action,
                'spikes': int(spikes.sum()),
                'n': n,
                'ep': ep,
                'episodes': args.episodes,
                'msg': msg if pygame.time.get_ticks() < msg_until else '',
                'msg_color': msg_color,
            }

            render(info, spikes)

            if args.no_truncate and step == 500:
                r.flash((30, 80, 50), ms=250)
                print('Past 500 steps — still balanced!', flush=True)

            fell = terminated
            capped = truncated and not args.no_truncate
            if fell or capped:
                past_cap = fell and step > 500
                solved = capped or (
                    fell and not past_cap and total >= args.max_steps
                )
                if fell:
                    label = (
                        f'FELL past cap — {int(total)} steps' if past_cap
                        else f'FELL — {int(total)} steps'
                    )
                    msg_color = BAD
                else:
                    label = f'SOLVED — {int(total)} steps'
                    msg_color = ACCENT
                msg = label
                msg_until = pygame.time.get_ticks() + 1500
                r.flash(
                    (120, 40, 40) if fell and not solved else (40, 120, 60),
                    ms=300,
                )

                end = pygame.time.get_ticks() + 1200
                while pygame.time.get_ticks() < end and running:
                    info['msg'] = msg
                    info['msg_color'] = msg_color
                    render(info, spikes)

                print(
                    f'Episode {ep + 1}: {int(total)} steps'
                    f'{" SOLVED" if solved else ""}',
                    flush=True,
                )
                ep += 1
                if ep >= args.episodes or not running:
                    break
                state, _ = env.reset(seed=args.seed + ep)
                neuron = LIFNeuron(n, decay=0.95, threshold=thr)
                step = 0
                total = 0.0
                msg = ''
            else:
                msg = ''
                msg_until = 0
    finally:
        env.close()
        if smap is not None:
            smap.close()
        pygame.quit()
    print('Done.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
