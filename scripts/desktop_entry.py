from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from xiyuan_mvp import __version__


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--self-test-ai",action="store_true")
    parser.add_argument("--acceptance-data",help="Directory containing the three packaged demonstration pairs")
    parser.add_argument("--offline",action="store_true",help="Block Python network connections during acceptance")
    parser.add_argument("--output",default="desktop-self-test.json")
    args=parser.parse_args()
    os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
    os.environ.setdefault("OMP_NUM_THREADS","4")
    bundled=Path(getattr(sys,'_MEIPASS',Path(__file__).parent))/'model_cache'
    if bundled.is_dir():
        os.environ['TORCH_HOME']=str(bundled)
    if args.self_test or args.acceptance_data:
        os.environ["QT_QPA_PLATFORM"]="offscreen"
    from PySide6.QtWidgets import QApplication
    from xiyuan_mvp.gui import MainWindow
    app=QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    window=MainWindow()
    if args.acceptance_data:
        from xiyuan_mvp.acceptance import run_workflow
        run_workflow(app,window,args.acceptance_data,args.output,offline=args.offline)
        return 0
    if args.self_test:
        import cv2
        import numpy as np
        from xiyuan_mvp.pipeline import StitchPipeline
        from xiyuan_mvp.run_io import write_json
        rng=np.random.default_rng(2026)
        scene=rng.integers(0,256,(360,900,3),np.uint8)
        for i in range(20):
            cv2.circle(scene,(30+i*40,40+(i*73)%280),10,(245,245,245),-1)
        result=StitchPipeline().run(scene[:,:580],scene[:,320:])
        if args.self_test_ai:
            from xiyuan_mvp.config import load_config
            config=load_config();config['inpainting']['engine']='neural_alignment'
            result=StitchPipeline(config).run(scene[:,:580],scene[:,320:],prepared=result,use_ai=True)
        window._on_result(result)
        window._set_busy(False)
        window.show()
        app.processEvents()
        screenshot=Path(args.output).with_suffix(".png")
        window.grab().save(str(screenshot))
        write_json(args.output,{"status":"passed","version":__version__,"frozen":bool(getattr(sys,"frozen",False)),
                                "image_size":list(result.final_image.shape),"gui_tabs":window.tabs.count(),
                                "metrics":result.metrics,"screenshot":str(screenshot)})
        window.close()
        return 0
    window.show()
    return app.exec()


if __name__=="__main__":
    try:
        raise SystemExit(main())
    except Exception:
        import traceback
        Path("desktop-crash.log").write_text(traceback.format_exc(),encoding="utf-8")
        raise
