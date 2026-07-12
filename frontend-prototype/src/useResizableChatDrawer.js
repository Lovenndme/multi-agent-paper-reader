import { useCallback, useEffect, useRef, useState } from "react";

const STORAGE_KEY = "paper-reader.chat-drawer-width";
const DEFAULT_WIDTH = 448;
const MIN_WIDTH = 400;
const MAX_WIDTH = 920;
const MOBILE_BREAKPOINT = 820;
const BACKGROUND_PEEK = 280;

function maximumWidthForViewport() {
  if (typeof window === "undefined") return MAX_WIDTH;
  return Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, window.innerWidth - BACKGROUND_PEEK));
}

function preferredWidth(value) {
  return Math.min(Math.max(Number(value) || DEFAULT_WIDTH, MIN_WIDTH), MAX_WIDTH);
}

function widthForViewport(value) {
  return Math.min(preferredWidth(value), maximumWidthForViewport());
}

function initialWidth() {
  if (typeof window === "undefined") return DEFAULT_WIDTH;
  const savedWidth = preferredWidth(window.localStorage.getItem(STORAGE_KEY));
  return window.innerWidth <= MOBILE_BREAKPOINT ? savedWidth : widthForViewport(savedWidth);
}

export function useResizableChatDrawer() {
  const [width, setWidth] = useState(initialWidth);
  const [maxWidth, setMaxWidth] = useState(maximumWidthForViewport);
  const [isResizing, setIsResizing] = useState(false);
  const widthRef = useRef(width);
  const preferredWidthRef = useRef(
    typeof window === "undefined" ? DEFAULT_WIDTH : preferredWidth(window.localStorage.getItem(STORAGE_KEY)),
  );
  const cleanupRef = useRef(null);

  const saveWidth = useCallback((nextWidth) => {
    const preferred = preferredWidth(nextWidth);
    const value = widthForViewport(preferred);
    preferredWidthRef.current = preferred;
    widthRef.current = value;
    setWidth(value);
    window.localStorage.setItem(STORAGE_KEY, String(preferred));
  }, []);

  useEffect(() => {
    function handleViewportResize() {
      const nextMax = maximumWidthForViewport();
      setMaxWidth(nextMax);
      if (window.innerWidth <= MOBILE_BREAKPOINT) return;
      const nextWidth = Math.min(preferredWidthRef.current, nextMax);
      widthRef.current = nextWidth;
      setWidth(nextWidth);
    }

    window.addEventListener("resize", handleViewportResize);
    return () => window.removeEventListener("resize", handleViewportResize);
  }, []);

  useEffect(() => () => cleanupRef.current?.(), []);

  const startResize = useCallback((event) => {
    if (event.button !== 0 || window.innerWidth <= MOBILE_BREAKPOINT) return;
    event.preventDefault();

    const startX = event.clientX;
    const startWidth = widthRef.current;
    const previousCursor = document.body.style.cursor;
    const previousUserSelect = document.body.style.userSelect;
    let finished = false;

    function finishResize() {
      if (finished) return;
      finished = true;
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", finishResize);
      window.removeEventListener("pointercancel", finishResize);
      document.body.style.cursor = previousCursor;
      document.body.style.userSelect = previousUserSelect;
      window.localStorage.setItem(STORAGE_KEY, String(preferredWidthRef.current));
      setIsResizing(false);
      cleanupRef.current = null;
    }

    function handlePointerMove(pointerEvent) {
      const value = widthForViewport(startWidth + startX - pointerEvent.clientX);
      preferredWidthRef.current = value;
      widthRef.current = value;
      setWidth(value);
    }

    setIsResizing(true);
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", finishResize);
    window.addEventListener("pointercancel", finishResize);
    cleanupRef.current = finishResize;
  }, []);

  const handleResizeKeyDown = useCallback((event) => {
    const step = event.shiftKey ? 80 : 32;
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      saveWidth(widthRef.current + step);
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      saveWidth(widthRef.current - step);
    } else if (event.key === "Home") {
      event.preventDefault();
      saveWidth(MIN_WIDTH);
    } else if (event.key === "End") {
      event.preventDefault();
      saveWidth(maximumWidthForViewport());
    }
  }, [saveWidth]);

  return {
    width,
    minWidth: MIN_WIDTH,
    maxWidth,
    isResizing,
    drawerStyle: { "--chat-drawer-width": `${width}px` },
    startResize,
    handleResizeKeyDown,
  };
}
