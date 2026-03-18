"use client";

import { useState, useRef, useCallback, useEffect } from "react";

// ─── Types ───────────────────────────────────────────────────────
type Point = { x: number; y: number };
type Step = 1 | 2 | 3 | 4;

// ─── Homography math (pure JS, no deps) ─────────────────────────

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
      for (let j = col; j <= n; j++) {
        M[row][j] -= factor * M[col][j];
      }
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

  const h = solveLinearSystem(A, b);
  return [...h, 1];
}

function applyHomography(H: number[], x: number, y: number): Point {
  const w = H[6] * x + H[7] * y + H[8];
  return {
    x: (H[0] * x + H[1] * y + H[2]) / w,
    y: (H[3] * x + H[4] * y + H[5]) / w,
  };
}

function bilinearInterpolate(
  data: Uint8ClampedArray,
  w: number,
  h: number,
  x: number,
  y: number
): [number, number, number, number] {
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(x0 + 1, w - 1);
  const y1 = Math.min(y0 + 1, h - 1);
  const fx = x - x0;
  const fy = y - y0;

  const idx00 = (y0 * w + x0) * 4;
  const idx10 = (y0 * w + x1) * 4;
  const idx01 = (y1 * w + x0) * 4;
  const idx11 = (y1 * w + x1) * 4;

  const result: [number, number, number, number] = [0, 0, 0, 0];
  for (let c = 0; c < 4; c++) {
    result[c] = Math.round(
      data[idx00 + c] * (1 - fx) * (1 - fy) +
        data[idx10 + c] * fx * (1 - fy) +
        data[idx01 + c] * (1 - fx) * fy +
        data[idx11 + c] * fx * fy
    );
  }
  return result;
}

// ─── Main Component ──────────────────────────────────────────────

export default function Home() {
  const [step, setStep] = useState<Step>(1);
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [imageName, setImageName] = useState("");
  const [points, setPoints] = useState<Point[]>([]);
  const [draggingIdx, setDraggingIdx] = useState<number | null>(null);
  const [width, setWidth] = useState("");
  const [height, setHeight] = useState("");
  const [resultDataUrl, setResultDataUrl] = useState<string | null>(null);
  const [resultInfo, setResultInfo] = useState<{
    w: number;
    h: number;
    dpi: number;
  } | null>(null);
  const [processing, setProcessing] = useState(false);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const loupeCanvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const canvasScale = 1;
  const canvasOffset: Point = { x: 0, y: 0 };

  // Loupe state
  const [loupeVisible, setLoupeVisible] = useState(false);
  const [loupePos, setLoupePos] = useState<Point>({ x: 0, y: 0 });

  // Drag precision: track start position for 0.5x dampening
  const dragStartScreen = useRef<Point | null>(null);
  const dragStartImg = useRef<Point | null>(null);

  // ─── Image loading ───────────────────────────────────────
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

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) loadImage(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith("image/")) loadImage(file);
  };

  // ─── Canvas helpers ──────────────────────────────────────
  function imgToCanvas(
    p: Point,
    scale: number,
    ox: number,
    oy: number
  ): Point {
    return { x: p.x * scale + ox, y: p.y * scale + oy };
  }

  function canvasToImg(
    cx: number,
    cy: number,
    scale: number,
    ox: number,
    oy: number
  ): Point {
    return { x: (cx - ox) / scale, y: (cy - oy) / scale };
  }

  function getCanvasTransform() {
    if (!canvasRef.current || !image || !containerRef.current) return null;
    const cw = containerRef.current.clientWidth;
    const ch = Math.min(
      containerRef.current.clientWidth * 1.1,
      window.innerHeight * 0.72
    );
    const scale =
      canvasScale * Math.min(cw / image.width, ch / image.height);
    const ox = canvasOffset.x + (cw - image.width * scale) / 2;
    const oy = canvasOffset.y + (ch - image.height * scale) / 2;
    return { scale, ox, oy, cw, ch };
  }

  // ─── Canvas drawing ──────────────────────────────────────
  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !image) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const container = containerRef.current;
    if (!container) return;

    const cw = container.clientWidth;
    const ch = Math.min(
      container.clientWidth * 1.1,
      window.innerHeight * 0.72
    );
    const dpr = window.devicePixelRatio;
    canvas.width = cw * dpr;
    canvas.height = ch * dpr;
    canvas.style.width = `${cw}px`;
    canvas.style.height = `${ch}px`;
    ctx.scale(dpr, dpr);

    ctx.fillStyle = "#0d1117";
    ctx.fillRect(0, 0, cw, ch);

    const scale =
      canvasScale * Math.min(cw / image.width, ch / image.height);
    const ox = canvasOffset.x + (cw - image.width * scale) / 2;
    const oy = canvasOffset.y + (ch - image.height * scale) / 2;

    ctx.drawImage(image, ox, oy, image.width * scale, image.height * scale);

    // Draw lines connecting points
    if (points.length > 0) {
      ctx.strokeStyle = "#e94560";
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      for (let i = 0; i < points.length; i++) {
        const p = imgToCanvas(points[i], scale, ox, oy);
        if (i === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
      }
      if (points.length === 4) ctx.closePath();
      ctx.stroke();
      ctx.setLineDash([]);

      // Semi-transparent fill
      if (points.length >= 3) {
        ctx.fillStyle = "rgba(233, 69, 96, 0.08)";
        ctx.beginPath();
        for (let i = 0; i < points.length; i++) {
          const p = imgToCanvas(points[i], scale, ox, oy);
          if (i === 0) ctx.moveTo(p.x, p.y);
          else ctx.lineTo(p.x, p.y);
        }
        ctx.closePath();
        ctx.fill();
      }

      // Draw point markers
      points.forEach((pt, idx) => {
        const p = imgToCanvas(pt, scale, ox, oy);
        ctx.beginPath();
        ctx.arc(p.x, p.y, 14, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(233, 69, 96, 0.3)";
        ctx.fill();
        ctx.beginPath();
        ctx.arc(p.x, p.y, 8, 0, Math.PI * 2);
        ctx.fillStyle = draggingIdx === idx ? "#ff6b81" : "#e94560";
        ctx.fill();
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.fillStyle = "#fff";
        ctx.font = "bold 11px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(`${idx + 1}`, p.x, p.y);
      });

      // Edge labels: show dimension mapping on edges
      if (points.length === 4) {
        const edgeLabels = [
          { from: 0, to: 1, label: "1\u21922  \u2194 sirina" },
          { from: 0, to: 3, label: "1\u21924  \u2195 visina" },
        ];
        edgeLabels.forEach(({ from, to, label }) => {
          const pA = imgToCanvas(points[from], scale, ox, oy);
          const pB = imgToCanvas(points[to], scale, ox, oy);
          const mx = (pA.x + pB.x) / 2;
          const my = (pA.y + pB.y) / 2;

          ctx.save();
          ctx.font = "bold 11px sans-serif";
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";

          // Background pill
          const tw = ctx.measureText(label).width + 12;
          ctx.fillStyle = "rgba(0,0,0,0.75)";
          ctx.beginPath();
          ctx.roundRect(mx - tw / 2, my - 10, tw, 20, 6);
          ctx.fill();

          ctx.fillStyle = "#ffd700";
          ctx.fillText(label, mx, my);
          ctx.restore();
        });
      }
    }

    // Instruction overlay at bottom
    if (points.length < 4) {
      ctx.fillStyle = "rgba(0,0,0,0.6)";
      ctx.fillRect(0, ch - 36, cw, 36);
      ctx.fillStyle = "#e0e0e0";
      ctx.font = "13px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(
        `Tapni/klikni točku ${points.length + 1} od 4`,
        cw / 2,
        ch - 18
      );
    }
  }, [image, points, canvasScale, canvasOffset, draggingIdx]);

  // ─── Loupe drawing ───────────────────────────────────────
  const drawLoupe = useCallback(
    (imgX: number, imgY: number, screenX: number, screenY: number) => {
      const loupeCanvas = loupeCanvasRef.current;
      if (!loupeCanvas || !image) return;
      const ctx = loupeCanvas.getContext("2d");
      if (!ctx) return;

      const loupeSize = 200;
      const zoomFactor = 5;
      const dpr = window.devicePixelRatio;

      loupeCanvas.width = loupeSize * dpr;
      loupeCanvas.height = loupeSize * dpr;
      loupeCanvas.style.width = `${loupeSize}px`;
      loupeCanvas.style.height = `${loupeSize}px`;
      ctx.scale(dpr, dpr);

      const srcSize = loupeSize / zoomFactor;
      const sx = imgX - srcSize / 2;
      const sy = imgY - srcSize / 2;

      ctx.clearRect(0, 0, loupeSize, loupeSize);

      // Clip to circle
      ctx.save();
      ctx.beginPath();
      ctx.arc(loupeSize / 2, loupeSize / 2, loupeSize / 2, 0, Math.PI * 2);
      ctx.clip();

      // Background
      ctx.fillStyle = "#0d1117";
      ctx.fillRect(0, 0, loupeSize, loupeSize);

      // Draw zoomed portion
      ctx.drawImage(
        image,
        sx,
        sy,
        srcSize,
        srcSize,
        0,
        0,
        loupeSize,
        loupeSize
      );

      // Crosshair
      ctx.strokeStyle = "rgba(233, 69, 96, 0.9)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(loupeSize / 2, 0);
      ctx.lineTo(loupeSize / 2, loupeSize);
      ctx.moveTo(0, loupeSize / 2);
      ctx.lineTo(loupeSize, loupeSize / 2);
      ctx.stroke();

      // Small center dot
      ctx.beginPath();
      ctx.arc(loupeSize / 2, loupeSize / 2, 3, 0, Math.PI * 2);
      ctx.fillStyle = "#e94560";
      ctx.fill();

      ctx.restore();

      // Border ring
      ctx.strokeStyle = "#e94560";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(
        loupeSize / 2,
        loupeSize / 2,
        loupeSize / 2 - 1.5,
        0,
        Math.PI * 2
      );
      ctx.stroke();

      // Position loupe above finger
      const container = containerRef.current;
      if (container) {
        const rect = container.getBoundingClientRect();
        let lx = screenX - rect.left - loupeSize / 2;
        let ly = screenY - rect.top - loupeSize - 60;

        lx = Math.max(4, Math.min(lx, rect.width - loupeSize - 4));
        if (ly < 4) ly = screenY - rect.top + 50;

        setLoupePos({ x: lx, y: ly });
      }
    },
    [image]
  );

  // ─── Touch/mouse handlers ────────────────────────────────
  const getEventPos = (
    e: React.TouchEvent | React.MouseEvent
  ): { clientX: number; clientY: number } | null => {
    if ("touches" in e) {
      if (e.touches.length > 0)
        return {
          clientX: e.touches[0].clientX,
          clientY: e.touches[0].clientY,
        };
      if (e.changedTouches.length > 0)
        return {
          clientX: e.changedTouches[0].clientX,
          clientY: e.changedTouches[0].clientY,
        };
      return null;
    }
    return { clientX: e.clientX, clientY: e.clientY };
  };

  const getCanvasPosFromClient = (
    clientX: number,
    clientY: number
  ): Point | null => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    return { x: clientX - rect.left, y: clientY - rect.top };
  };

  const findNearPoint = (canvasPos: Point, threshold: number = 28): number => {
    const t = getCanvasTransform();
    if (!t) return -1;
    for (let i = 0; i < points.length; i++) {
      const p = imgToCanvas(points[i], t.scale, t.ox, t.oy);
      const dist = Math.hypot(canvasPos.x - p.x, canvasPos.y - p.y);
      if (dist < threshold) return i;
    }
    return -1;
  };

  const handlePointerDown = (e: React.TouchEvent | React.MouseEvent) => {
    if (!image) return;
    const pos = getEventPos(e);
    if (!pos) return;
    const { clientX, clientY } = pos;
    const canvasPos = getCanvasPosFromClient(clientX, clientY);
    if (!canvasPos) return;

    const nearIdx = findNearPoint(canvasPos);
    if (nearIdx >= 0) {
      setDraggingIdx(nearIdx);
      setLoupeVisible(true);
      // Store start positions for dampened drag
      dragStartScreen.current = { x: clientX, y: clientY };
      dragStartImg.current = { x: points[nearIdx].x, y: points[nearIdx].y };
      const t = getCanvasTransform()!;
      const imgPos = canvasToImg(
        canvasPos.x,
        canvasPos.y,
        t.scale,
        t.ox,
        t.oy
      );
      drawLoupe(imgPos.x, imgPos.y, clientX, clientY);
      e.preventDefault();
      return;
    }

    if (points.length < 4) {
      const t = getCanvasTransform()!;
      const imgPos = canvasToImg(
        canvasPos.x,
        canvasPos.y,
        t.scale,
        t.ox,
        t.oy
      );
      const clampedX = Math.max(0, Math.min(imgPos.x, image.width - 1));
      const clampedY = Math.max(0, Math.min(imgPos.y, image.height - 1));
      const newPoints = [...points, { x: clampedX, y: clampedY }];
      setPoints(newPoints);

      if (newPoints.length === 4) {
        setStep(3);
      }
    }
  };

  const handlePointerMove = (e: React.TouchEvent | React.MouseEvent) => {
    if (draggingIdx === null || !image) return;
    e.preventDefault();
    const pos = getEventPos(e);
    if (!pos) return;
    const { clientX, clientY } = pos;

    // Dampened drag: 0.5x speed for precision
    const DAMPEN = 0.5;
    const t = getCanvasTransform()!;

    if (dragStartScreen.current && dragStartImg.current) {
      const dxScreen = clientX - dragStartScreen.current.x;
      const dyScreen = clientY - dragStartScreen.current.y;

      // Convert screen delta to image delta, then apply dampening
      const imgDx = (dxScreen / t.scale) * DAMPEN;
      const imgDy = (dyScreen / t.scale) * DAMPEN;

      const newX = Math.max(
        0,
        Math.min(dragStartImg.current.x + imgDx, image.width - 1)
      );
      const newY = Math.max(
        0,
        Math.min(dragStartImg.current.y + imgDy, image.height - 1)
      );

      const newPoints = [...points];
      newPoints[draggingIdx] = { x: newX, y: newY };
      setPoints(newPoints);

      drawLoupe(newX, newY, clientX, clientY);
    }
  };

  const handlePointerUp = () => {
    setDraggingIdx(null);
    setLoupeVisible(false);
    dragStartScreen.current = null;
    dragStartImg.current = null;
  };

  // Redraw canvas on state changes
  useEffect(() => {
    drawCanvas();
  }, [drawCanvas]);

  useEffect(() => {
    const handleResize = () => drawCanvas();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [drawCanvas]);

  // Native touch listeners with { passive: false } to prevent page scroll
  const pointsRef = useRef(points);
  pointsRef.current = points;
  const draggingIdxRef = useRef(draggingIdx);
  draggingIdxRef.current = draggingIdx;
  const imageRef = useRef(image);
  imageRef.current = image;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const onTouchStart = (e: TouchEvent) => {
      e.preventDefault();
    };
    const onTouchMove = (e: TouchEvent) => {
      e.preventDefault();
    };

    canvas.addEventListener("touchstart", onTouchStart, { passive: false });
    canvas.addEventListener("touchmove", onTouchMove, { passive: false });
    return () => {
      canvas.removeEventListener("touchstart", onTouchStart);
      canvas.removeEventListener("touchmove", onTouchMove);
    };
  }, [step, image]);

  // ─── Perspective correction ──────────────────────────────
  const runCorrection = useCallback(() => {
    if (!image || points.length !== 4 || !width || !height) return;

    setProcessing(true);

    requestAnimationFrame(() => {
      setTimeout(() => {
        try {
          const realW = parseFloat(width);
          const realH = parseFloat(height);
          if (realW <= 0 || realH <= 0) {
            alert("Unesite pozitivne dimenzije.");
            setProcessing(false);
            return;
          }

          const srcPoints = points;

          // Estimate pixel density from source quadrilateral
          const avgSrcWidth =
            (Math.hypot(
              srcPoints[1].x - srcPoints[0].x,
              srcPoints[1].y - srcPoints[0].y
            ) +
              Math.hypot(
                srcPoints[2].x - srcPoints[3].x,
                srcPoints[2].y - srcPoints[3].y
              )) /
            2;
          const avgSrcHeight =
            (Math.hypot(
              srcPoints[3].x - srcPoints[0].x,
              srcPoints[3].y - srcPoints[0].y
            ) +
              Math.hypot(
                srcPoints[2].x - srcPoints[1].x,
                srcPoints[2].y - srcPoints[1].y
              )) /
            2;

          const pxPerCmW = avgSrcWidth / realW;
          const pxPerCmH = avgSrcHeight / realH;
          const pxPerCm = Math.max(pxPerCmW, pxPerCmH);

          let outW = Math.round(realW * pxPerCm);
          let outH = Math.round(realH * pxPerCm);

          // Cap to avoid browser crash
          const maxDim = 8000;
          if (outW > maxDim || outH > maxDim) {
            const capScale = maxDim / Math.max(outW, outH);
            outW = Math.round(outW * capScale);
            outH = Math.round(outH * capScale);
          }

          // Destination corners
          const dst: Point[] = [
            { x: 0, y: 0 },
            { x: outW, y: 0 },
            { x: outW, y: outH },
            { x: 0, y: outH },
          ];

          // Homography: dst → src (map output pixels back to source)
          const H = computeHomography(dst, srcPoints);

          // Get source pixel data
          const srcCanvas = document.createElement("canvas");
          srcCanvas.width = image.width;
          srcCanvas.height = image.height;
          const srcCtx = srcCanvas.getContext("2d")!;
          srcCtx.drawImage(image, 0, 0);
          const srcData = srcCtx.getImageData(
            0,
            0,
            image.width,
            image.height
          );

          // Create output canvas
          const outCanvas = document.createElement("canvas");
          outCanvas.width = outW;
          outCanvas.height = outH;
          const outCtx = outCanvas.getContext("2d")!;
          const outImageData = outCtx.createImageData(outW, outH);

          // Transform pixel by pixel with bilinear interpolation
          for (let y = 0; y < outH; y++) {
            for (let x = 0; x < outW; x++) {
              const srcPt = applyHomography(H, x, y);
              const outIdx = (y * outW + x) * 4;

              if (
                srcPt.x >= 0 &&
                srcPt.x < image.width - 1 &&
                srcPt.y >= 0 &&
                srcPt.y < image.height - 1
              ) {
                const [r, g, b, a] = bilinearInterpolate(
                  srcData.data,
                  image.width,
                  image.height,
                  srcPt.x,
                  srcPt.y
                );
                outImageData.data[outIdx] = r;
                outImageData.data[outIdx + 1] = g;
                outImageData.data[outIdx + 2] = b;
                outImageData.data[outIdx + 3] = a;
              }
            }
          }

          outCtx.putImageData(outImageData, 0, 0);

          const effectiveDpi = Math.round((outW / realW) * 2.54);

          setResultDataUrl(outCanvas.toDataURL("image/png"));
          setResultInfo({ w: outW, h: outH, dpi: effectiveDpi });
          setStep(4);
        } catch (err) {
          alert("Greška pri korekciji: " + (err as Error).message);
        } finally {
          setProcessing(false);
        }
      }, 50);
    });
  }, [image, points, width, height]);

  // ─── Download ────────────────────────────────────────────
  const handleDownload = () => {
    if (!resultDataUrl) return;
    const a = document.createElement("a");
    a.href = resultDataUrl;
    const baseName = imageName.replace(/\.[^.]+$/, "");
    a.download = `${baseName}_korigirano.png`;
    a.click();
  };

  // ─── Reset ───────────────────────────────────────────────
  const reset = () => {
    setStep(1);
    setImage(null);
    setImageName("");
    setPoints([]);
    setWidth("");
    setHeight("");
    setResultDataUrl(null);
    setResultInfo(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const resetPoints = () => {
    setPoints([]);
    setStep(2);
  };

  // ─── Step labels ─────────────────────────────────────────
  const stepLabels = [
    { n: 1, label: "Učitaj" },
    { n: 2, label: "Označi" },
    { n: 3, label: "Dimenzije" },
    { n: 4, label: "Rezultat" },
  ];

  return (
    <main className="flex flex-col items-center min-h-screen px-3 py-4 max-w-2xl mx-auto">
      {/* Header */}
      <h1 className="text-xl font-bold tracking-tight mb-1 font-mono text-foreground">
        PerspektivKor <span className="text-xs font-normal text-text-dim">v0.3</span>
      </h1>
      <p className="text-xs text-text-dim mb-4">
        Perspektivna korekcija fotografija
      </p>

      {/* Progress steps */}
      <div className="flex items-center gap-1 mb-6 w-full max-w-xs">
        {stepLabels.map((s, i) => (
          <div key={s.n} className="flex items-center flex-1">
            <div className="flex flex-col items-center flex-1">
              <div
                className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold transition-colors ${
                  step >= s.n
                    ? "bg-accent text-white"
                    : "bg-surface-light text-text-dim"
                }`}
              >
                {step > s.n ? "\u2713" : s.n}
              </div>
              <span className="text-[10px] mt-1 text-text-dim">
                {s.label}
              </span>
            </div>
            {i < stepLabels.length - 1 && (
              <div
                className={`h-0.5 flex-1 mx-1 mb-4 transition-colors ${
                  step > s.n ? "bg-accent" : "bg-surface-light"
                }`}
              />
            )}
          </div>
        ))}
      </div>

      {/* ─── Step 1: Upload ─── */}
      {step === 1 && (
        <div
          className="w-full border-2 border-dashed border-surface-light rounded-xl p-8 text-center cursor-pointer hover:border-accent transition-colors"
          onClick={() => fileInputRef.current?.click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            capture="environment"
            className="hidden"
            onChange={handleFileChange}
          />
          <div className="text-5xl mb-3 opacity-50">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="w-14 h-14 mx-auto text-text-dim"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={1.5}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M6.827 6.175A2.31 2.31 0 015.186 7.23c-.38.054-.757.112-1.134.175C2.999 7.58 2.25 8.507 2.25 9.574V18a2.25 2.25 0 002.25 2.25h15A2.25 2.25 0 0021.75 18V9.574c0-1.067-.75-1.994-1.802-2.169a47.865 47.865 0 00-1.134-.175 2.31 2.31 0 01-1.64-1.055l-.822-1.316a2.192 2.192 0 00-1.736-1.039 48.774 48.774 0 00-5.232 0 2.192 2.192 0 00-1.736 1.039l-.821 1.316z"
              />
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M16.5 12.75a4.5 4.5 0 11-9 0 4.5 4.5 0 019 0z"
              />
            </svg>
          </div>
          <p className="text-foreground font-medium mb-1">
            Tapni za kameru ili odaberi fotku
          </p>
          <p className="text-xs text-text-dim">
            Drag &amp; drop ili klikni za galeriju
          </p>
        </div>
      )}

      {/* ─── Step 2-3: Mark points + Dimensions ─── */}
      {step >= 2 && step <= 3 && image && (
        <div className="w-full">
          <div
            ref={containerRef}
            className="relative w-full rounded-lg overflow-hidden bg-[#0d1117] touch-canvas"
          >
            <canvas
              ref={canvasRef}
              onMouseDown={handlePointerDown}
              onMouseMove={handlePointerMove}
              onMouseUp={handlePointerUp}
              onMouseLeave={handlePointerUp}
              onTouchStart={handlePointerDown}
              onTouchMove={handlePointerMove}
              onTouchEnd={handlePointerUp}
              className="w-full cursor-crosshair"
            />

            {/* Zoom loupe - appears when dragging a point */}
            {loupeVisible && (
              <canvas
                ref={loupeCanvasRef}
                className="absolute pointer-events-none rounded-full"
                style={{
                  left: loupePos.x,
                  top: loupePos.y,
                  width: 200,
                  height: 200,
                  boxShadow: "0 4px 24px rgba(0,0,0,0.6)",
                }}
              />
            )}
          </div>

          {/* Point controls */}
          <div className="flex gap-2 mt-3 justify-center">
            {points.length > 0 && (
              <button
                onClick={resetPoints}
                className="px-4 py-2 bg-surface-light text-foreground rounded-lg text-sm hover:bg-surface-light/80 transition-colors"
              >
                Resetiraj
              </button>
            )}
            {points.length === 4 && step === 2 && (
              <button
                onClick={() => setStep(3)}
                className="px-4 py-2 bg-accent text-white rounded-lg text-sm font-medium hover:bg-accent-hover transition-colors"
              >
                Dalje
              </button>
            )}
          </div>
        </div>
      )}

      {/* ─── Step 3: Dimensions input ─── */}
      {step === 3 && (
        <div className="w-full mt-4 bg-surface rounded-xl p-5">
          <h3 className="text-sm font-medium text-foreground mb-3">
            Stvarne dimenzije pravokutnika
          </h3>
          <div className="flex gap-3 items-end">
            <div className="flex-1">
              <label className="text-xs text-text-dim block mb-1">
                Sirina 1→2 (cm)
              </label>
              <input
                type="number"
                value={width}
                onChange={(e) => setWidth(e.target.value)}
                placeholder="100"
                className="w-full bg-surface-light text-foreground rounded-lg px-3 py-2.5 font-mono text-sm outline-none focus:ring-2 focus:ring-accent"
                inputMode="decimal"
              />
            </div>
            <span className="text-text-dim pb-2.5 font-mono text-lg">
              x
            </span>
            <div className="flex-1">
              <label className="text-xs text-text-dim block mb-1">
                Visina 1→4 (cm)
              </label>
              <input
                type="number"
                value={height}
                onChange={(e) => setHeight(e.target.value)}
                placeholder="70"
                className="w-full bg-surface-light text-foreground rounded-lg px-3 py-2.5 font-mono text-sm outline-none focus:ring-2 focus:ring-accent"
                inputMode="decimal"
              />
            </div>
          </div>
          <button
            onClick={runCorrection}
            disabled={!width || !height || processing}
            className="w-full mt-4 py-3 bg-accent text-white rounded-lg font-medium text-sm hover:bg-accent-hover transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {processing ? "Procesiranje..." : "Korigiraj perspektivu"}
          </button>
        </div>
      )}

      {/* ─── Step 4: Result ─── */}
      {step === 4 && resultDataUrl && (
        <div className="w-full">
          <div className="w-full rounded-lg overflow-hidden bg-[#0d1117]">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={resultDataUrl}
              alt="Korigirana slika"
              className="w-full h-auto"
            />
          </div>
          {resultInfo && (
            <div className="mt-3 bg-surface rounded-lg p-3 text-xs font-mono text-text-dim flex gap-4 justify-center">
              <span>
                {resultInfo.w} x {resultInfo.h} px
              </span>
              <span>~{resultInfo.dpi} DPI</span>
            </div>
          )}
          <div className="flex gap-2 mt-3">
            <button
              onClick={handleDownload}
              className="flex-1 py-3 bg-success text-black rounded-lg font-medium text-sm hover:opacity-90 transition-opacity"
            >
              Download PNG
            </button>
            <button
              onClick={reset}
              className="flex-1 py-3 bg-surface-light text-foreground rounded-lg text-sm hover:opacity-90 transition-opacity"
            >
              Nova slika
            </button>
          </div>
        </div>
      )}

      {/* Back link */}
      {step > 1 && step < 4 && (
        <button
          onClick={reset}
          className="mt-4 text-xs text-text-dim hover:text-foreground transition-colors"
        >
          ← Nova slika
        </button>
      )}
    </main>
  );
}
