import { useCallback, useEffect, useRef, useState } from "react";


const BOTTOM_THRESHOLD = 72;


export function useChatAutoScroll(messages, isStreaming, conversationKey) {
  const containerRef = useRef(null);
  const previousCountRef = useRef(0);
  const [autoFollow, setAutoFollow] = useState(true);

  const scrollToBottom = useCallback((behavior = "smooth") => {
    const container = containerRef.current;
    if (!container) return;
    setAutoFollow(true);
    container.scrollTo({ top: container.scrollHeight, behavior });
  }, []);

  useEffect(() => {
    setAutoFollow(true);
    previousCountRef.current = messages.length;
    window.requestAnimationFrame(() => scrollToBottom("auto"));
  }, [conversationKey, scrollToBottom]);

  useEffect(() => {
    const previousCount = previousCountRef.current;
    const addedMessages = messages.slice(previousCount);
    previousCountRef.current = messages.length;
    if (addedMessages.some((message) => message.role === "user")) {
      window.requestAnimationFrame(() => scrollToBottom("smooth"));
      return;
    }
    if (!autoFollow) return;
    window.requestAnimationFrame(() => {
      const container = containerRef.current;
      if (container) container.scrollTop = container.scrollHeight;
    });
  }, [messages, isStreaming, autoFollow, scrollToBottom]);

  const handleScroll = useCallback(() => {
    const container = containerRef.current;
    if (!container) return;
    const distanceFromBottom = container.scrollHeight - container.scrollTop - container.clientHeight;
    setAutoFollow(distanceFromBottom <= BOTTOM_THRESHOLD);
  }, []);

  return {
    containerRef,
    autoFollow,
    handleScroll,
    scrollToBottom,
  };
}
