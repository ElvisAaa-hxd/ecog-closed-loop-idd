# -*- coding: utf-8 -*-
"""Headless video export of stream_fft_judge.py.

Runs the exact streaming/processing loop of the judge script with the
SDL dummy video driver, captures every rendered frame and writes an H.264
MP4 at 30 fps (real-time pace). Constants are parsed from the judge script
itself so the video always reflects the shipped algorithm parameters.

Output: closed_loop_stream_new.mp4 (in the repository working root)
        data_output_video.txt  (same 5-column log as the judge script)
"""
import os
import re

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import pygame
from collections import deque
from itertools import islice
import numpy as np
import imageio.v2 as imageio

BASE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE)
JUDGE = os.path.join(BASE, "stream_fft_judge.py")
OUT_VIDEO = os.path.join(REPO_ROOT, "closed_loop_stream_new.mp4")
OUT_LOG = os.path.join(BASE, "data_output_video.txt")

# ---- constants: single source of truth = the judge script ----
src = open(JUDGE, encoding="utf-8").read()
g = lambda name: float(re.search(r"^%s\s*=\s*([0-9.]+)" % name, src, re.M).group(1))
SEIZ_D = g("SEIZ_D")
SEIZ_T = int(g("SEIZ_T"))
SEIZ_R = g("SEIZ_R")
SHORT_WINDOW = int(g("SHORT_WINDOW"))
LONG_WINDOW = int(g("LONG_WINDOW"))
THRESH_LOCK_S = g("THRESH_LOCK_S")
BP_F0 = g("BP_F0")
BP_Q = g("BP_Q")
print("params from judge.py: D=%g T=%d R=%g SHORT=%d LONG=%d LOCK=%g F0=%g Q=%g"
      % (SEIZ_D, SEIZ_T, SEIZ_R, SHORT_WINDOW, LONG_WINDOW, THRESH_LOCK_S, BP_F0, BP_Q))

WHITE = (255, 255, 255)

data_line = 250
variance_line = 400
state_line = 650
cure_line = 700

xdata = deque(range(100, 1601))
ydata = deque([data_line] * 1000, maxlen=1500)
state = deque([state_line] * 1000, maxlen=1500)
cure = deque([cure_line + 100] * 1000, maxlen=1500)
variance_plot = deque([variance_line] * 1500, maxlen=1500)
values_queue = deque(maxlen=SHORT_WINDOW)
short_buf = deque(maxlen=SHORT_WINDOW)
long_buf = deque(maxlen=LONG_WINDOW)
fft_queue = deque(maxlen=600)

Epilepsy_judge = 0
Epilepsy_cure = 0
judge_until = 0.0
cure_until = 0.0
thresh_locked = False
lock_until = 0.0
frozen_thresh = 0.0
seizure_window_start = 0.0
seizure_count = 0

with open(os.path.join(REPO_ROOT, "data", "EPILEPSY.csv"), "r") as file:
    lines = list(islice(file, 3, None))

head_t = [float(line.split(",")[0]) for line in lines[:300]]
dt = float(np.median(np.diff(head_t)))
fs = 1.0 / dt
w0 = 2.0 * np.pi * BP_F0 / fs
c0 = np.cos(w0)
s0 = np.sin(w0)
alpha = s0 / (2.0 * BP_Q)
a0 = 1.0 + alpha
bp_b0 = alpha / a0
bp_b2 = -alpha / a0
bp_a1 = (-2.0 * c0) / a0
bp_a2 = (1.0 - alpha) / a0
bp_z1 = 0.0
bp_z2 = 0.0

FFT_X0, FFT_X1, FFT_Y = 100, 700, 900
FFT_FMAX = 50.0
FFT_TICK_HZ = 10
FPS = 30
FRAME_S = 1.0 / FPS

pygame.init()
screen = pygame.display.set_mode((2000, 1000))
pygame.display.set_caption("FFT judge (Brain_ECoG)")
clock = pygame.time.Clock()
font = pygame.font.SysFont("arial", 28)
tick_font = pygame.font.SysFont("arial", 20)
text_surface = font.render("Waiting", True, WHITE)
detect_count = 0
variance = 0.0

writer = imageio.get_writer(
    OUT_VIDEO, fps=FPS, codec="libx264", quality=8,
    macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"],
)

print("Begin to streaming")
k = 0
n = len(lines)
nframes = 0
try:
    with open(OUT_LOG, "w") as f:
        while k < n:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise SystemExit

            t_end = float(lines[k].split(",")[0]) + FRAME_S
            while k < n and float(lines[k].split(",")[0]) < t_end:
                values = lines[k].split(",")
                k += 1
                value = float(values[1])
                time_value = float(values[0])

                ydata.appendleft(value + data_line)

                bp_y = bp_b0 * value + bp_z1
                bp_z1 = -bp_a1 * bp_y + bp_z2
                bp_z2 = bp_b2 * value - bp_a2 * bp_y
                short_buf.append(bp_y)
                long_buf.append(bp_y)

                if time_value - seizure_window_start >= 1.0:
                    seizure_count = 0
                    seizure_window_start = time_value

                if len(short_buf) >= SHORT_WINDOW and len(long_buf) >= LONG_WINDOW:
                    if thresh_locked and time_value >= lock_until:
                        thresh_locked = False
                    thresh_live = SEIZ_D * np.var(long_buf)
                    thresh_use = frozen_thresh if thresh_locked else thresh_live
                    S = np.var(short_buf)
                    if S > thresh_use:
                        if not thresh_locked:
                            frozen_thresh = thresh_live
                            thresh_locked = True
                            lock_until = time_value + THRESH_LOCK_S
                        if Epilepsy_cure == 0:
                            seizure_count += 1
                            if seizure_count >= SEIZ_T:
                                dur = SEIZ_R * S * SHORT_WINDOW
                                if dur < 0.1:
                                    dur = 0.1
                                if dur > 10.0:
                                    dur = 10.0
                                Epilepsy_cure = 1
                                cure_until = time_value + dur
                                seizure_count = 0
                                detect_count += 1
                                text_surface = font.render("Detected %d" % detect_count, True, WHITE)
                                Epilepsy_judge = 1
                                judge_until = time_value + 0.1
                                print("t=%.3f s  therapy=%.3f s" % (time_value, dur))

                if Epilepsy_cure == 1 and time_value >= cure_until:
                    Epilepsy_cure = 0
                    seizure_count = 0

                if Epilepsy_judge == 1 and time_value < judge_until:
                    MCU_Data = -100
                else:
                    Epilepsy_judge = 0
                    MCU_Data = 0
                state.appendleft(MCU_Data + state_line)

                if Epilepsy_cure == 1 and time_value < cure_until:
                    cure_state_data = 0
                else:
                    Epilepsy_cure = 0
                    cure_state_data = 100
                cure.appendleft(cure_state_data + cure_line)

                values_queue.append(value)
                if len(values_queue) > 1:
                    variance = np.var(values_queue)
                    variance_plot.appendleft(variance * 0.1 + variance_line)

                f.write(
                    "%s\t%s\t%s\t%s\t%s\n"
                    % (
                        round(time_value, 3),
                        round(value, 3),
                        round(variance, 3),
                        MCU_Data * -1,
                        cure_state_data * -1 + 100,
                    )
                )
                fft_queue.append(value)

            screen.fill((0, 0, 0))
            screen.blit(text_surface, (100, 100))
            pygame.draw.lines(screen, (255, 255, 255), False, list(zip(xdata, ydata)), 1)
            pygame.draw.lines(screen, (0, 0, 255), False, list(zip(xdata, state)), 1)
            pygame.draw.lines(screen, (255, 0, 0), False, list(zip(xdata, cure)), 1)
            pygame.draw.lines(screen, (255, 255, 0), False, list(zip(xdata, variance_plot)), 1)
            freq = 0
            while freq <= FFT_FMAX + 1e-6:
                x = int(FFT_X0 + freq / FFT_FMAX * (FFT_X1 - FFT_X0))
                label = tick_font.render("%d" % int(round(freq)), True, WHITE)
                screen.blit(label, (x - label.get_width() // 2, FFT_Y + 4))
                freq += FFT_TICK_HZ
            screen.blit(tick_font.render("Hz", True, WHITE), (FFT_X1 + 28, FFT_Y + 4))
            if len(fft_queue) == 600:
                spec = np.abs(np.fft.rfft(np.asarray(fft_queue)))
                freqs = np.fft.rfftfreq(600, d=dt)
                points = [
                    (FFT_X0 + freqs[i] / FFT_FMAX * (FFT_X1 - FFT_X0), FFT_Y - spec[i] / 25)
                    for i in range(len(spec))
                    if freqs[i] <= FFT_FMAX
                ]
                pygame.draw.lines(screen, (255, 255, 255), False, points, 1)
            pygame.display.flip()

            frame = pygame.surfarray.array3d(screen)  # (W,H,3) RGB
            writer.append_data(np.transpose(frame, (1, 0, 2)))
            nframes += 1
            clock.tick(FPS)
finally:
    writer.close()
    pygame.quit()

print("frames written: %d | video: %s" % (nframes, OUT_VIDEO))
