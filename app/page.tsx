"use client";

import { useState, useRef, useCallback, useEffect } from "react";

type Point = { x: number; y: number };
type Step = 1 | 2 | 3 | 4;

// ─── Homography math ─────────────────────────────────────────────

function solveLinearSystem(A: number[][], b: number[]): number[] {
  const n = A.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let maxRow = col;
    let maxVal = Math.abs(M[col][col]);
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(M[row][col]) > maxVal) {
        maxVal = Math.abs(M[row][col]);
        maxRow = row;
      }
    }
    [M[col], M[maxRow]] = [M[maxRow], M[col]];
    const pivot = M[col][col];
    if (Math.abs(pivot) < 1e-12) throw new Error("Singular matrix");
    for (let j = col; j <= n; j++) M[col][j] /= pivot;
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = M[row][col];
      for (let j = col; j <= n; j++) M[row][j] -= factor * M[col][j];
    }
  }
  return M.map((row) => row[n]);
}

function computeHomography(src: Point[], dst: Point[]): number[] {
  const A: number[][] = [];
  const b: number[] = [];
  for (let i = 0; i < 4; i++) {
    const { x: sx, y: sy } = src[i];
    const { x: dx, y: dy } = dst[i];
    A.push([sx, sy, 1, 0, 0, 0, -dx * sx, -dx * sy]);
    b.push(dx);
    A.push([0, 0, 0, sx, sy, 1, -dy * sx, -dy * sy]);
    b.push(dy);
  }
  return [...solveLinearSystem(A, b), 1];
}

function applyHomography(H: number[], x: number, y: number): Point {
  const w = H[6] * x + H[7] * y + H[8];
  return {
    x: (H[0] * x + H[1] * y + H[2]) / w,
    y: (H[3] * x + H[4] * y + H[5]) / w,
  };
}

function bilinearInterpolate(
  data: Uint8ClampedArray, w: number, h: number, x: number, y: number
): [number, number, number, number] {
  const x0 = Math.floor(x), y0 = Math.floor(y);
  const x1 = Math.min(x0 + 1, w - 1), y1 = Math.min(y0 + 1, h - 1);
  const fx = x - x0, fy = y - y0;
  const i00 = (y0 * w + x0) * 4, i10 = (y0 * w + x1) * 4;
  const i01 = (y1 * w + x0) * 4, i11 = (y1 * w + x1) * 4;
  const r: [number, number, number, number] = [0, 0, 0, 0];
  for (let c = 0; c < 4; c++) {
    r[c] = Math.round(
      data[i00 + c] * (1 - fx) * (1 - fy) + data[i10 + c] * fx * (1 - fy) +
      data[i01 + c] * (1 - fx) * fy + data[i11 + c] * fx * fy
    );
  }
  return r;
}

// ─── Component ───────────────────────────────────────────────────

export default function Home() {
  const [step, setStep] = useState<Step>(1);
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [imageName, setImageName] = useState("");
  const [points, setPoints] = useState<Point[]>([]);
  const [draggingIdx, setDraggingIdx] = useState<number | null>(null);
  const [widthCm, setWidthCm] = useState("");
  const [heightCm, setHeightCm] = useState("");
  const [resultDataUrl, setResultDataUrl] = useState<string | null>(null);
  const [resultInfo, setResultInfo] = useState<{ w: number; h: number; dpi: number } | null>(null);
  const [processing, setProcessing] = useState(false);
  const [loupeVisible, setLoupeVisible] = useState(false);
  const [loupePos, setLoupePos] = useState<Point>({ x: 0, y: 0 });

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const loupeRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const dragStartScreen = useRef<Point | null>(null);
  const dragStartImg = useRef<Point | null>(null);

  // Use refs for state accessed in native event handlers
  const pointsRef = useRef(points);
  pointsRef.current = points;
  const draggingIdxRef = useRef(draggingIdx);
  draggingIdxRef.current = draggingIdx;
  const imageRef = useRef(image);
  imageRef.current = image;

  // ─── Image loading ─────────────────────────────────────────
  const loadImage = useCallback((file: File) => {
    setImageName(file.name);
    const img = new Image();
    img.onload = () => {
      setImage(img);
      setPoints([]);
      setStep(2);
      setResultDataUrl(null);
      setResultInfo(null);
    };
    img.src = URL.createObjectURL(file);
  }, []);

  // ─── Canvas transform (uses actual rendered size) ──────────
  const getTransform = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imageRef.current;
    if (!canvas || !img) return null;
    const rect = canvas.getBoundingClientRect();
    const cw = rect.width;
    const ch = rect.height;
    const scale = Math.min(cw / img.width, ch / img.height);
    const ox = (cw - img.width * scale) / 2;
    const oy = (ch - img.height * scale) / 2;
    return { scale, ox, oy, cw, ch };
  }, []);

  const imgToCanvas = (p: Point, s: number, ox: number, oy: number): Point =>
    ({ x: p.x * s + ox, y: p.y * s + oy });

  const canvasToImg = (cx: number, cy: number, s: number, ox: number, oy: number): Point =>
    ({ x: (cx - ox) / s, y: (cy - oy) / s });

  // ─── Canvas drawing ────────────────────────────────────────
  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !image || !container) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const cw = container.clientWidth;
    const ch = container.clientHeight;
    if (ch <= 0) return;

    const dpr = window.devicePixelRatio;
    canvas.width = cw * dpr;
    canvas.height = ch * dpr;
    canvas.style.width = `${cw}px`;
    canvas.style.height = `${ch}px`;
    ctx.scale(dpr, dpr);

    ctx.fillStyle = "#0d1117";
    ctx.fillRect(0, 0, cw, ch);

    const scale = Math.min(cw / image.width, ch / image.height);
    const ox = (cw - image.width * scale) / 2;
    const oy = (ch - image.height * scale) / 2;

    ctx.drawImage(image, ox, oy, image.width * scale, image.height * scale);

    if (points.length > 0) {
      // Lines
      ctx.strokeStyle = "#e94560";
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      for (let i = 0; i < points.length; i++) {
        const p = imgToCanvas(points[i], scale, ox, oy);
        if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
      }
      if (points.length === 4) ctx.closePath();
      ctx.stroke();
      ctx.setLineDash([]);

      if (points.length >= 3) {
        ctx.fillStyle = "rgba(233, 69, 96, 0.08)";
        ctx.beginPath();
        for (let i = 0; i < points.length; i++) {
          const p = imgToCanvas(points[i], scale, ox, oy);
          if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
        }
        ctx.closePath();
        ctx.fill();
      }

      // Point markers
      points.forEach((pt, idx) => {
        const p = imgToCanvas(pt, scale, ox, oy);
        ctx.beginPath(); ctx.arc(p.x, p.y, 14, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(233, 69, 96, 0.3)"; ctx.fill();
        ctx.beginPath(); ctx.arc(p.x, p.y, 8, 0, Math.PI * 2);
        ctx.fillStyle = draggingIdx === idx ? "#ff6b81" : "#e94560"; ctx.fill();
        ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.stroke();
        ctx.fillStyle = "#fff"; ctx.font = "bold 11px sans-serif";
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText(`${idx + 1}`, p.x, p.y);
      });

      // Edge labels
      if (points.length === 4) {
        [
          { from: 0, to: 1, label: "1\u21922 \u2194 sirina" },
          { from: 0, to: 3, label: "1\u21924 \u2195 visina" },
        ].forEach(({ from, to, label }) => {
          const pA = imgToCanvas(points[from], scale, ox, oy);
          const pB = imgToCanvas(points[to], scale, ox, oy);
          const mx = (pA.x + pB.x) / 2, my = (pA.y + pB.y) / 2;
          ctx.save();
          ctx.font = "bold 11px sans-serif";
          ctx.textAlign = "center"; ctx.textBaseline = "middle";
          const tw = ctx.measureText(label).width + 12;
          ctx.fillStyle = "rgba(0,0,0,0.75)";
          ctx.beginPath(); ctx.roundRect(mx - tw / 2, my - 10, tw, 20, 6); ctx.fill();
          ctx.fillStyle = "#ffd700"; ctx.fillText(label, mx, my);
          ctx.restore();
        });
      }
    }

    if (points.length < 4) {
      ctx.fillStyle = "rgba(0,0,0,0.6)";
      ctx.fillRect(0, ch - 32, cw, 32);
      ctx.fillStyle = "#e0e0e0"; ctx.font = "13px sans-serif";
      ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText(`Tapni točku ${points.length + 1} od 4`, cw / 2, ch - 16);
    }
  }, [image, points, draggingIdx]);

  // ─── Loupe ─────────────────────────────────────────────────
  const drawLoupe = useCallback((imgX: number, imgY: number, screenX: number, screenY: number) => {
    const lc = loupeRef.current;
    const img = imageRef.current;
    if (!lc || !img) return;
    const ctx = lc.getContext("2d");
    if (!ctx) return;

    const S = 200, Z = 5, dpr = window.devicePixelRatio;
    lc.width = S * dpr; lc.height = S * dpr;
    lc.style.width = `${S}px`; lc.style.height = `${S}px`;
    ctx.scale(dpr, dpr);

    const src = S / Z;
    ctx.save();
    ctx.beginPath(); ctx.arc(S / 2, S / 2, S / 2, 0, Math.PI * 2); ctx.clip();
    ctx.fillStyle = "#0d1117"; ctx.fillRect(0, 0, S, S);
    ctx.drawImage(img, imgX - src / 2, imgY - src / 2, src, src, 0, 0, S, S);
    ctx.strokeStyle = "rgba(233,69,96,0.9)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(S / 2, 0); ctx.lineTo(S / 2, S);
    ctx.moveTo(0, S / 2); ctx.lineTo(S, S / 2); ctx.stroke();
    ctx.beginPath(); ctx.arc(S / 2, S / 2, 3, 0, Math.PI * 2);
    ctx.fillStyle = "#e94560"; ctx.fill();
    ctx.restore();
    ctx.strokeStyle = "#e94560"; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.arc(S / 2, S / 2, S / 2 - 1.5, 0, Math.PI * 2); ctx.stroke();

    const container = containerRef.current;
    if (container) {
      const rect = container.getBoundingClientRect();
      let lx = screenX - rect.left - S / 2;
      let ly = screenY - rect.top - S - 50;
      lx = Math.max(4, Math.min(lx, rect.width - S - 4));
      if (ly < 4) ly = screenY - rect.top + 40;
      setLoupePos({ x: lx, y: ly });
    }
  }, []);

  // ─── Pointer helpers ───────────────────────────────────────
  const getTouchPos = (e: TouchEvent): Point | null => {
    const t = e.touches[0] || e.changedTouches[0];
    return t ? { x: t.clientX, y: t.clientY } : null;
  };

  const getCanvasLocal = (clientX: number, clientY: number): Point | null => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const r = canvas.getBoundingClientRect();
    return { x: clientX - r.left, y: clientY - r.top };
  };

  const findNear = useCallback((cx: number, cy: number): number => {
    const t = getTransform();
    if (!t) return -1;
    const pts = pointsRef.current;
    for (let i = 0; i < pts.length; i++) {
      const p = imgToCanvas(pts[i], t.scale, t.ox, t.oy);
      if (Math.hypot(cx - p.x, cy - p.y) < 30) return i;
    }
    return -1;
  }, [getTransform]);

  // ─── Native touch handlers (passive:false) ─────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !image) return;

    const onDown = (e: TouchEvent) => {
      e.preventDefault();
      const pos = getTouchPos(e);
      if (!pos) return;
      const local = getCanvasLocal(pos.x, pos.y);
      if (!local) return;

      const near = findNear(local.x, local.y);
      if (near >= 0) {
        draggingIdxRef.current = near;
        setDraggingIdx(near);
        setLoupeVisible(true);
        dragStartScreen.current = pos;
        dragStartImg.current = { x: pointsRef.current[near].x, y: pointsRef.current[near].y };
        drawLoupe(pointsRef.current[near].x, pointsRef.current[near].y, pos.x, pos.y);
        return;
      }

      if (pointsRef.current.length < 4) {
        const t = getTransform();
        if (!t) return;
        const ip = canvasToImg(local.x, local.y, t.scale, t.ox, t.oy);
        const img = imageRef.current!;
        const nx = Math.max(0, Math.min(ip.x, img.width - 1));
        const ny = Math.max(0, Math.min(ip.y, img.height - 1));
        const newPts = [...pointsRef.current, { x: nx, y: ny }];
        setPoints(newPts);
        if (newPts.length === 4) setStep(3);
      }
    };

    const onMove = (e: TouchEvent) => {
      e.preventDefault();
      if (draggingIdxRef.current === null) return;
      const pos = getTouchPos(e);
      if (!pos || !dragStartScreen.current || !dragStartImg.current) return;

      const t = getTransform();
      if (!t) return;
      const img = imageRef.current!;
      const dx = (pos.x - dragStartScreen.current.x) / t.scale * 0.25;
      const dy = (pos.y - dragStartScreen.current.y) / t.scale * 0.25;
      const nx = Math.max(0, Math.min(dragStartImg.current.x + dx, img.width - 1));
      const ny = Math.max(0, Math.min(dragStartImg.current.y + dy, img.height - 1));

      const newPts = [...pointsRef.current];
      newPts[draggingIdxRef.current] = { x: nx, y: ny };
      setPoints(newPts);
      drawLoupe(nx, ny, pos.x, pos.y);
    };

    const onUp = (e: TouchEvent) => {
      e.preventDefault();
      draggingIdxRef.current = null;
      setDraggingIdx(null);
      setLoupeVisible(false);
      dragStartScreen.current = null;
      dragStartImg.current = null;
    };

    canvas.addEventListener("touchstart", onDown, { passive: false });
    canvas.addEventListener("touchmove", onMove, { passive: false });
    canvas.addEventListener("touchend", onUp, { passive: false });
    return () => {
      canvas.removeEventListener("touchstart", onDown);
      canvas.removeEventListener("touchmove", onMove);
      canvas.removeEventListener("touchend", onUp);
    };
  }, [image, step, findNear, getTransform, drawLoupe]);

  // ─── Mouse handlers (desktop) ──────────────────────────────
  const handleMouseDown = (e: React.MouseEvent) => {
    if (!image) return;
    const local = getCanvasLocal(e.clientX, e.clientY);
    if (!local) return;

    const near = findNear(local.x, local.y);
    if (near >= 0) {
      setDraggingIdx(near);
      setLoupeVisible(true);
      dragStartScreen.current = { x: e.clientX, y: e.clientY };
      dragStartImg.current = { x: points[near].x, y: points[near].y };
      drawLoupe(points[near].x, points[near].y, e.clientX, e.clientY);
      return;
    }

    if (points.length < 4) {
      const t = getTransform();
      if (!t) return;
      const ip = canvasToImg(local.x, local.y, t.scale, t.ox, t.oy);
      const nx = Math.max(0, Math.min(ip.x, image.width - 1));
      const ny = Math.max(0, Math.min(ip.y, image.height - 1));
      const newPts = [...points, { x: nx, y: ny }];
      setPoints(newPts);
      if (newPts.length === 4) setStep(3);
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (draggingIdx === null || !image) return;
    const t = getTransform();
    if (!t || !dragStartScreen.current || !dragStartImg.current) return;

    const dx = (e.clientX - dragStartScreen.current.x) / t.scale * 0.25;
    const dy = (e.clientY - dragStartScreen.current.y) / t.scale * 0.25;
    const nx = Math.max(0, Math.min(dragStartImg.current.x + dx, image.width - 1));
    const ny = Math.max(0, Math.min(dragStartImg.current.y + dy, image.height - 1));

    const newPts = [...points];
    newPts[draggingIdx] = { x: nx, y: ny };
    setPoints(newPts);
    drawLoupe(nx, ny, e.clientX, e.clientY);
  };

  const handleMouseUp = () => {
    setDraggingIdx(null);
    setLoupeVisible(false);
    dragStartScreen.current = null;
    dragStartImg.current = null;
  };

  // ─── Effects ───────────────────────────────────────────────
  useEffect(() => { drawCanvas(); }, [drawCanvas]);
  useEffect(() => {
    const h = () => drawCanvas();
    window.addEventListener("resize", h);
    return () => window.removeEventListener("resize", h);
  }, [drawCanvas]);

  // ─── Correction ────────────────────────────────────────────
  const runCorrection = useCallback(() => {
    if (!image || points.length !== 4 || !widthCm || !heightCm) return;
    setProcessing(true);
    requestAnimationFrame(() => setTimeout(() => {
      try {
        const rW = parseFloat(widthCm), rH = parseFloat(heightCm);
        if (rW <= 0 || rH <= 0) { alert("Pozitivne dimenzije!"); setProcessing(false); return; }

        const sp = points;
        const avgW = (Math.hypot(sp[1].x-sp[0].x,sp[1].y-sp[0].y)+Math.hypot(sp[2].x-sp[3].x,sp[2].y-sp[3].y))/2;
        const avgH = (Math.hypot(sp[3].x-sp[0].x,sp[3].y-sp[0].y)+Math.hypot(sp[2].x-sp[1].x,sp[2].y-sp[1].y))/2;
        const ppc = Math.max(avgW/rW, avgH/rH);
        let oW = Math.round(rW*ppc), oH = Math.round(rH*ppc);
        const mx = 8000;
        if (oW > mx || oH > mx) { const s = mx/Math.max(oW,oH); oW = Math.round(oW*s); oH = Math.round(oH*s); }

        const dst: Point[] = [{x:0,y:0},{x:oW,y:0},{x:oW,y:oH},{x:0,y:oH}];
        const H = computeHomography(dst, sp);

        const sc = document.createElement("canvas"); sc.width = image.width; sc.height = image.height;
        const sctx = sc.getContext("2d")!; sctx.drawImage(image, 0, 0);
        const sd = sctx.getImageData(0, 0, image.width, image.height);

        const oc = document.createElement("canvas"); oc.width = oW; oc.height = oH;
        const octx = oc.getContext("2d")!; const od = octx.createImageData(oW, oH);

        for (let y = 0; y < oH; y++) for (let x = 0; x < oW; x++) {
          const p = applyHomography(H, x, y);
          const i = (y * oW + x) * 4;
          if (p.x >= 0 && p.x < image.width-1 && p.y >= 0 && p.y < image.height-1) {
            const [r,g,b,a] = bilinearInterpolate(sd.data, image.width, image.height, p.x, p.y);
            od.data[i]=r; od.data[i+1]=g; od.data[i+2]=b; od.data[i+3]=a;
          }
        }
        octx.putImageData(od, 0, 0);
        setResultDataUrl(oc.toDataURL("image/png"));
        setResultInfo({ w: oW, h: oH, dpi: Math.round((oW/rW)*2.54) });
        setStep(4);
      } catch (err) { alert("Greška: " + (err as Error).message); }
      finally { setProcessing(false); }
    }, 50));
  }, [image, points, widthCm, heightCm]);

  const handleDownload = () => {
    if (!resultDataUrl) return;
    const a = document.createElement("a");
    a.href = resultDataUrl;
    a.download = `${imageName.replace(/\.[^.]+$/, "")}_korigirano.png`;
    a.click();
  };

  const reset = () => {
    setStep(1); setImage(null); setImageName(""); setPoints([]);
    setWidthCm(""); setHeightCm(""); setResultDataUrl(null); setResultInfo(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const steps = [
    { n: 1, l: "Učitaj" }, { n: 2, l: "Označi" },
    { n: 3, l: "Dimenzije" }, { n: 4, l: "Rezultat" },
  ];

  return (
    <main className="flex flex-col items-center h-[100dvh] overflow-hidden px-3 py-3 max-w-2xl mx-auto">
      {/* Header */}
      <h1 className="text-lg font-bold tracking-tight font-mono text-foreground">
        PerspektivKor <span className="text-xs font-normal text-text-dim">v0.9</span>
      </h1>

      {/* Progress */}
      <div className="flex items-center gap-1 my-2 w-full max-w-xs">
        {steps.map((s, i) => (
          <div key={s.n} className="flex items-center flex-1">
            <div className="flex flex-col items-center flex-1">
              <div className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold ${step >= s.n ? "bg-accent text-white" : "bg-surface-light text-text-dim"}`}>
                {step > s.n ? "\u2713" : s.n}
              </div>
              <span className="text-[9px] mt-0.5 text-text-dim">{s.l}</span>
            </div>
            {i < 3 && <div className={`h-0.5 flex-1 mx-1 mb-3 ${step > s.n ? "bg-accent" : "bg-surface-light"}`} />}
          </div>
        ))}
      </div>

      {/* Step 1 */}
      {step === 1 && (
        <div
          className="w-full flex flex-col items-center justify-center border-2 border-dashed border-surface-light rounded-xl p-10 cursor-pointer hover:border-accent transition-colors"
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => { e.preventDefault(); const f = e.dataTransfer.files?.[0]; if (f?.type.startsWith("image/")) loadImage(f); }}
        >
          <input ref={fileInputRef} type="file" accept="image/*" capture="environment" className="hidden"
            onChange={(e) => { const f = e.target.files?.[0]; if (f) loadImage(f); }} />
          <svg xmlns="http://www.w3.org/2000/svg" className="w-12 h-12 text-text-dim mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M6.827 6.175A2.31 2.31 0 015.186 7.23c-.38.054-.757.112-1.134.175C2.999 7.58 2.25 8.507 2.25 9.574V18a2.25 2.25 0 002.25 2.25h15A2.25 2.25 0 0021.75 18V9.574c0-1.067-.75-1.994-1.802-2.169a47.865 47.865 0 00-1.134-.175 2.31 2.31 0 01-1.64-1.055l-.822-1.316a2.192 2.192 0 00-1.736-1.039 48.774 48.774 0 00-5.232 0 2.192 2.192 0 00-1.736 1.039l-.821 1.316z" />
            <path strokeLinecap="round" strokeLinejoin="round" d="M16.5 12.75a4.5 4.5 0 11-9 0 4.5 4.5 0 019 0z" />
          </svg>
          <p className="text-foreground font-medium text-sm">Tapni za kameru ili odaberi fotku</p>
          <p className="text-xs text-text-dim mt-1">Drag &amp; drop ili klikni za galeriju</p>
        </div>
      )}

      {/* Step 2-3: Canvas + controls */}
      {step >= 2 && step <= 3 && image && (
        <div className="w-full flex-1 flex flex-col min-h-0 gap-2">
          {/* Canvas container - takes available space */}
          <div ref={containerRef} className="relative w-full flex-1 min-h-0 rounded-lg overflow-hidden bg-[#0d1117]" style={{ touchAction: "none" }}>
            <canvas
              ref={canvasRef}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
              className="absolute inset-0 w-full h-full cursor-crosshair"
              style={{ touchAction: "none" }}
            />
            {loupeVisible && (
              <canvas ref={loupeRef} className="absolute pointer-events-none rounded-full"
                style={{ left: loupePos.x, top: loupePos.y, width: 200, height: 200, boxShadow: "0 4px 24px rgba(0,0,0,0.6)" }} />
            )}
          </div>

          {/* Controls below canvas */}
          <div className="flex gap-2 justify-center shrink-0">
            {points.length > 0 && (
              <button onClick={() => { setPoints([]); setStep(2); }} className="px-3 py-1.5 bg-surface-light text-foreground rounded-lg text-sm">Resetiraj</button>
            )}
            {points.length === 4 && step === 2 && (
              <button onClick={() => setStep(3)} className="px-3 py-1.5 bg-accent text-white rounded-lg text-sm font-medium">Dalje</button>
            )}
          </div>

          {/* Dimension inputs (step 3) */}
          {step === 3 && (
            <div className="bg-surface rounded-xl p-4 shrink-0">
              <div className="flex gap-3 items-end">
                <div className="flex-1">
                  <label className="text-xs text-text-dim block mb-1">Sirina 1→2 (cm)</label>
                  <input type="number" value={widthCm} onChange={(e) => setWidthCm(e.target.value)}
                    placeholder="upisi cm" inputMode="decimal"
                    className="w-full bg-surface-light text-foreground rounded-lg px-3 py-2 font-mono text-sm outline-none focus:ring-2 focus:ring-accent" />
                </div>
                <span className="text-text-dim pb-2 font-mono">x</span>
                <div className="flex-1">
                  <label className="text-xs text-text-dim block mb-1">Visina 1→4 (cm)</label>
                  <input type="number" value={heightCm} onChange={(e) => setHeightCm(e.target.value)}
                    placeholder="upisi cm" inputMode="decimal"
                    className="w-full bg-surface-light text-foreground rounded-lg px-3 py-2 font-mono text-sm outline-none focus:ring-2 focus:ring-accent" />
                </div>
              </div>
              <button onClick={runCorrection} disabled={!widthCm || !heightCm || processing}
                className="w-full mt-3 py-2.5 bg-success text-black rounded-lg font-medium text-sm disabled:opacity-40">
                {processing ? "Procesiranje..." : "Korigiraj perspektivu"}
              </button>
            </div>
          )}

          <button onClick={reset} className="text-xs text-text-dim shrink-0">← Nova slika</button>
        </div>
      )}

      {/* Step 4: Result */}
      {step === 4 && resultDataUrl && (
        <div className="w-full flex-1 flex flex-col min-h-0 gap-2">
          <div className="flex-1 min-h-0 overflow-y-auto rounded-lg bg-[#0d1117]">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={resultDataUrl} alt="Korigirana slika" className="w-full h-auto" />
          </div>
          {resultInfo && (
            <div className="bg-surface rounded-lg p-2 text-xs font-mono text-text-dim flex gap-4 justify-center shrink-0">
              <span>{resultInfo.w} x {resultInfo.h} px</span>
              <span>~{resultInfo.dpi} DPI</span>
            </div>
          )}
          <div className="flex gap-2 shrink-0">
            <button onClick={handleDownload} className="flex-1 py-2.5 bg-success text-black rounded-lg font-medium text-sm">Download PNG</button>
            <button onClick={reset} className="flex-1 py-2.5 bg-surface-light text-foreground rounded-lg text-sm">Nova slika</button>
          </div>
        </div>
      )}
    </main>
  );
}
