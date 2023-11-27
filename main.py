
import view
import tracking
import sys

flag_view = False
flag_track = False
flag_fft = False
flag_model = False

if len(sys.argv) > 1:
    if "--view" in sys.argv:
        flag_view = True
    if "--track" in sys.argv:
        flag_track = True
    if "--fft" in sys.argv:
        flag_fft = True
    if "--model" in sys.argv:
        flag_model = True
else:
    print("Usage: python main.py [--view] [--track] [--fft] [--model] ./18-nov/18nov-libre plastique-0g0-0aimants.avi")
    exit(0)

if(not sys.argv[-1].startswith("--")):
    FILE = sys.argv[-1]
else:
    FILE = "18-nov/18nov-libre plastique-0g0-0aimants.avi"

if flag_track:
    tracking.track(FILE)

if flag_view:
    view.viewData(FILE, flag_model, flag_fft)
