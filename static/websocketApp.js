// static/websocketApp.js

/* global marked */

// --- Configuration ---
const MAX_RECONNECT_ATTEMPTS = 5;
const RECONNECT_BASE_DELAY_MS = 5000;
const ROBOT_ICON_PATH = "robot1.jpg";
const THINKING_INDICATOR_ID = "thinking-indicator-wrapper";
const VOICE_WEBSOCKET_ENDPOINT = "/ws/voice/";
const TARGET_SAMPLE_RATE = 16000;

// --- Module State ---
let textWs = null;
let voiceWs = null;
let reconnectAttempts = 0;
let messageForm, messageInput, messagesDiv, sendButton, micButton, micIcon;
let appInitialized = false;

let audioContext;
let mediaStream;
let audioProcessorNode;
let isListening = false;
let textWsReconnectAttempts = 0; // Separate counter for text WS

// --- NEW: Audio Playback Queue and State ---
let browserAudioPlaybackQueue = []; // Array to hold ArrayBuffer chunks
let isPlayingAudio = false;       // Flag to indicate if playback is active
let expectedPlaybackSampleRate = 24000; // Audio from Gemini is 24kHz
let isCurrentlyPlayingFromServer = false; // More descriptive flag

// --- Helper Functions ---
function addStatusMessage(text, typeClass) {
  if (!messagesDiv) { console.error("UI_LOG: Cannot add status message: messagesDiv not found."); return; }
  console.log(`UI_LOG: addStatusMessage - Text: "${text}", Class: "${typeClass}"`);
  try {
    const p = document.createElement("p");
    p.classList.add("system-status-message");
    const span = document.createElement("span");
    span.className = typeClass;
    span.textContent = text;
    p.appendChild(span);
    messagesDiv.appendChild(p);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
  } catch (e) { console.error("UI_LOG: Error adding status message:", e); }
}

function showThinkingIndicator(isVoice = false) {
    hideThinkingIndicator();
    if (!messagesDiv) { console.error("UI_LOG: Cannot show thinking indicator: messagesDiv not found."); return;}
    const wrapper = document.createElement("div");
    wrapper.id = THINKING_INDICATOR_ID;
    wrapper.classList.add("message-wrapper", "thinking");
    const iconSpan = document.createElement("span");
    iconSpan.classList.add("message-icon", "robot-icon");
    const robotImg = document.createElement("img");
    robotImg.src = ROBOT_ICON_PATH; robotImg.alt = "Agent icon";
    iconSpan.appendChild(robotImg);
    const bubbleP = document.createElement("p");
    bubbleP.classList.add("message-bubble", "thinking-bubble");
    const thinkingText = isVoice ? "Listening..." : "Thinking";
    bubbleP.innerHTML = `${thinkingText}<span class="dots"><span>.</span><span>.</span><span>.</span></span>`;
    wrapper.appendChild(iconSpan);
    wrapper.appendChild(bubbleP);
    messagesDiv.appendChild(wrapper);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
    console.log(isVoice ? "UI_LOG: Showing listening indicator." : "UI_LOG: Showing thinking indicator.");
}

function hideThinkingIndicator() {
  const indicator = document.getElementById(THINKING_INDICATOR_ID);
  if (indicator) {
    indicator.remove();
    console.log("UI_LOG: Hiding thinking/listening indicator.");
  }
}

function addMessageToUI(messageText, senderType, isMarkdown = true) {
    if (!messagesDiv) { console.error("UI_LOG: Cannot add message: messagesDiv not found."); return; }
    console.log(`UI_LOG: addMessageToUI - Sender: ${senderType}, Text: "${messageText.substring(0, 50)}..."`);
    hideThinkingIndicator(); 
    const wrapper = document.createElement("div"); // ... (rest of your addMessageToUI)
    wrapper.classList.add("message-wrapper", senderType);
    const iconSpan = document.createElement("span");
    iconSpan.classList.add("message-icon");
    const bubbleP = document.createElement("p");
    bubbleP.classList.add("message-bubble");

    if (senderType === "user") {
        iconSpan.classList.add("user-icon");
        iconSpan.innerHTML = `<span class="material-icons-outlined">person</span>`;
        bubbleP.classList.add("user-message");
        bubbleP.textContent = messageText;
    } else { 
        iconSpan.classList.add("robot-icon");
        const robotImg = document.createElement("img");
        robotImg.src = ROBOT_ICON_PATH; robotImg.alt = "Agent icon";
        iconSpan.appendChild(robotImg);
        bubbleP.classList.add("server-message-block");
        if (isMarkdown && typeof marked !== "undefined") {
            try { bubbleP.innerHTML = marked.parse(messageText); }
            catch (e) { console.error("UI_LOG: Error parsing Markdown:", e); bubbleP.textContent = messageText;}
        } else { bubbleP.textContent = messageText; }
    }
    wrapper.appendChild(iconSpan);
    wrapper.appendChild(bubbleP);
    messagesDiv.appendChild(wrapper);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

// --- Text WebSocket Handlers ---
function handleTextWebSocketOpen(event) {
  console.log("UI_LOG: Text WebSocket connection opened:", event.target.url);
  textWsReconnectAttempts = 0; // Reset counter on successful connection
  if (sendButton) sendButton.disabled = false;
  if (micButton) micButton.disabled = false;
  addStatusMessage("Text chat connected", "connection-open-text");
  addSubmitHandler();
}
function handleTextWebSocketMessage(event) { /* ... as before ... */ 
    console.log("UI_LOG: TextWS received message:", event.data);
    hideThinkingIndicator();
    try {
        const packet = JSON.parse(event.data);
        if (packet.message) { addMessageToUI(packet.message, "server"); }
        else { console.warn("UI_LOG: Received text packet without 'message':", packet); }
    } catch (e) {
        console.error("UI_LOG: Error parsing text WS message:", e, "Raw:", event.data);
        addStatusMessage(`Error processing server text: ${e.message}`, "error-text");
        addMessageToUI(`Received non-JSON: ${event.data}`, "server", false);
    }
}
function handleTextWebSocketClose(event) { /* ... as before ... */ 
    console.warn(`UI_LOG: Text WebSocket closed. Code: ${event.code}, Clean: ${event.wasClean}`);
    hideThinkingIndicator();
    if (sendButton) sendButton.disabled = true;
    removeSubmitHandler();
    // Only attempt reconnect if it wasn't a clean closure by the client/app explicitly
    // and we haven't exceeded attempts.
    // Code 1000 is normal closure. Other codes might indicate issues.
    // For simplicity, let's try to reconnect on most non-1000 closures or if not explicitly closed by UI.
    if (event.code !== 1000 && textWsReconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
        textWsReconnectAttempts++;
        const reconnectDelay = Math.min(
            30000,
            RECONNECT_BASE_DELAY_MS * Math.pow(2, textWsReconnectAttempts - 1)
        );
        addStatusMessage(
            `Text chat disconnected. Attempting reconnect ${textWsReconnectAttempts}/${MAX_RECONNECT_ATTEMPTS} in ${Math.round(reconnectDelay / 1000)}s...`,
            "connection-closed-text"
        );
        setTimeout(connectTextWebSocket, reconnectDelay); // Attempt to reconnect
    } else if (textWsReconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
        console.error("UI_LOG: Max text WebSocket reconnection attempts reached.");
        addStatusMessage(
            "Text chat connection lost permanently. Please reload the page.",
            "error-text"
        );
    } else {
        // Clean closure or max attempts not an issue yet, just show disconnected.
        addStatusMessage("Text chat disconnected.", "connection-closed-text");
    }
    textWs = null; // Clear the instance
}
function handleTextWebSocketError(error) { /* ... as before ... */ 
    console.error("UI_LOG: Text WebSocket error:", error);
    hideThinkingIndicator();
    addStatusMessage("Text WebSocket connection error.", "error-text");
}
function addTextWebSocketHandlers(wsInstance) { /* ... as before ... */ 
    wsInstance.onopen = handleTextWebSocketOpen;
    wsInstance.onmessage = handleTextWebSocketMessage;
    wsInstance.onclose = handleTextWebSocketClose;
    wsInstance.onerror = handleTextWebSocketError;
}
function connectTextWebSocket() {
    // If already connecting or open, don't try again immediately
    if (textWs && (textWs.readyState === WebSocket.OPEN || textWs.readyState === WebSocket.CONNECTING)) {
        console.log("UI_LOG: Text WebSocket already open or connecting.");
        return;
    }

    const sessionId = `text-${Math.random().toString(36).substring(2, 10)}`;
    const wsProtocol = window.location.protocol === "https:" ? "wss://" : "ws://";
    const wsUrl = wsProtocol + window.location.host + "/ws/" + sessionId; 
    console.log(`UI_LOG: Attempting Text WebSocket connect: ${wsUrl} (Attempt: ${textWsReconnectAttempts + 1})`);
    try {
        // Ensure any previous instance is fully closed or nulled before creating new
        if (textWs) {
            textWs.onopen = null; textWs.onmessage = null; textWs.onclose = null; textWs.onerror = null;
            if (textWs.readyState !== WebSocket.CLOSED) {
                 textWs.close(1000, "Client re-initiating connection");
            }
        }
        textWs = new WebSocket(wsUrl);
        addTextWebSocketHandlers(textWs);
    } catch (error) { 
        console.error("UI_LOG: Error creating Text WS instance:", error);
        // This catch might not be effective for all WebSocket constructor errors.
        // The onerror handler is more reliable for connection failures.
        if (handleTextWebSocketClose) { // Check if function exists to avoid error during setup
            handleTextWebSocketClose({ code: 1006, reason: "Failed to create WebSocket instance", wasClean: false });
        }
    }
}

// --- Voice WebSocket Handlers ---
function handleVoiceWebSocketOpen(event) {
    console.log("UI_LOG: Voice WebSocket connection opened:", event.target.url);
    addStatusMessage("Voice channel connected", "connection-open-text");
    if (micButton) micButton.disabled = false;
    if (micIcon) micIcon.textContent = "mic";
    if (micButton) micButton.classList.remove("is-listening");
}


async function handleVoiceWebSocketMessage(event) {
    console.log("UI_LOG: VoiceWS received data - Type:", typeof event.data, "Is ArrayBuffer:", event.data instanceof ArrayBuffer, "Is Blob:", event.data instanceof Blob);
    if (event.data instanceof ArrayBuffer || event.data instanceof Blob) {
        const byteLength = event.data.byteLength || event.data.size;
        console.log(`UI_LOG: VoiceWS is processing BINARY data: ${byteLength} bytes`);
        const audioData = event.data instanceof Blob ? await event.data.arrayBuffer() : event.data;
        if (byteLength > 0) {
             browserAudioPlaybackQueue.push(audioData); // Add chunk to queue
             console.log(`UI_LOG: Added ${byteLength} bytes to playback queue. Queue size: ${browserAudioPlaybackQueue.length}`);
             if (!isCurrentlyPlayingFromServer) { // Only start if not already playing
                 playNextChunkFromQueue();
             }               // Attempt to play if not already playing
        } else {
            console.warn("UI_LOG: VoiceWS received empty binary data chunk.");
        }
    } else if (typeof event.data === 'string') {
        // ... (your existing JSON parsing for gemini_transcript etc.) ...
        // (This part for text messages like transcripts seems fine)
        console.log("UI_LOG: VoiceWS received TEXT data:", event.data);
        try {
            const packet = JSON.parse(event.data);
            console.log("UI_LOG: VoiceWS parsed JSON packet:", packet);
            if (packet.type === "gemini_transcript") {
                console.log("UI_LOG: Gemini Transcript to display:", packet.text);
                hideThinkingIndicator(); 
                addMessageToUI(packet.text, "server"); 
            } else if (packet.type === "user_transcript") {
                console.log("UI_LOG: User Transcript:", packet.text);
                // Potentially display this as a user message if desired
                addMessageToUI(packet.text, "user"); 
                showThinkingIndicator(false); 
            } else if (packet.type === "error") {
                console.error("UI_LOG: Error from voice server:", packet.message);
                addStatusMessage(`Voice Agent Error: ${packet.message}`, "error-text");
                stopListening();
            } else {
                console.log("UI_LOG: Received unknown text packet from voice server:", packet);
            }
            
        } catch (e) { console.error("UI_LOG: Error parsing JSON from voice server:", e); }
    } else {
        console.warn("UI_LOG: VoiceWS received data of unexpected type:", typeof event.data);
    }
}


function handleVoiceWebSocketClose(event) {
    console.warn(`UI_LOG: Voice WebSocket closed. Code: ${event.code}, Clean: ${event.wasClean}, Reason: ${event.reason}`);
    stopListening();
    if (micButton) micButton.disabled = false; // Allow user to try reconnecting by clicking mic
    addStatusMessage("Voice channel disconnected.", "connection-closed-text");
    voiceWs = null;
}

function handleVoiceWebSocketError(error) {
    console.error("UI_LOG: Voice WebSocket error:", error);
    stopListening();
    if (micButton) micButton.disabled = false;
    addStatusMessage("Voice WebSocket connection error.", "error-text");
    voiceWs = null;
}

function addVoiceWebSocketHandlers(wsInstance) {
    wsInstance.binaryType = "arraybuffer";
    wsInstance.onopen = handleVoiceWebSocketOpen;
    wsInstance.onmessage = handleVoiceWebSocketMessage;
    wsInstance.onclose = handleVoiceWebSocketClose;
    wsInstance.onerror = handleVoiceWebSocketError;
    console.log("UI_LOG: Voice WebSocket event handlers attached.");
}

function connectVoiceWebSocket() {
    if (voiceWs && voiceWs.readyState === WebSocket.OPEN) { /* ... as before ... */ return Promise.resolve(voiceWs); }
    if (voiceWs && voiceWs.readyState === WebSocket.CONNECTING) { /* ... as before ... */ 
        return new Promise((resolve, reject) => { /* ... */ });
    }
    const sessionId = `voice-${Math.random().toString(36).substring(2, 10)}`;
    const wsProtocol = window.location.protocol === "https:" ? "wss://" : "ws://";
    const wsUrl = wsProtocol + window.location.host + VOICE_WEBSOCKET_ENDPOINT + sessionId;
    console.log(`UI_LOG: Attempting Voice WebSocket connect: ${wsUrl}`);
    return new Promise((resolve, reject) => {
        try {
            voiceWs = new WebSocket(wsUrl);
            addVoiceWebSocketHandlers(voiceWs);
            const originalOnOpen = voiceWs.onopen;
            voiceWs.onopen = (event) => { if (originalOnOpen) originalOnOpen(event); resolve(voiceWs); };
            const originalOnError = voiceWs.onerror;
            voiceWs.onerror = (event) => { if (originalOnError) originalOnError(event); reject(new Error("Voice WS connection failed"));};
        } catch (error) { /* ... */ reject(error); }
    });
}

// --- Web Audio API Logic ---
async function startMicrophone() {
    console.log("UI_LOG: startMicrophone called.");
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) { /* ... */ return null; }
    try {
        if (!audioContext) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            console.log("UI_LOG: AudioContext created. Initial state:", audioContext.state);
        }
        if (audioContext.state === 'suspended') {
            console.log("UI_LOG: AudioContext is suspended, attempting to resume...");
            await audioContext.resume();
            console.log("UI_LOG: AudioContext resumed. Current state:", audioContext.state);
        }

        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
        console.log("UI_LOG: Microphone access granted.");
        const source = audioContext.createMediaStreamSource(mediaStream);
        const bufferSize = 4096; // ~250ms at 16kHz, ~85ms at 48kHz. A common value.
        audioProcessorNode = audioContext.createScriptProcessor(bufferSize, 1, 1);
        
        audioProcessorNode.onaudioprocess = (audioProcessingEvent) => {
            if (!isListening || !voiceWs || voiceWs.readyState !== WebSocket.OPEN) return;
            const inputData = audioProcessingEvent.inputBuffer.getChannelData(0);
            // console.log("UI_LOG: onaudioprocess - Raw input data length:", inputData.length); // Very verbose
            const downsampledData = downsampleBuffer(inputData, audioContext.sampleRate, TARGET_SAMPLE_RATE);
            const int16Pcm = convertFloat32ToInt16(downsampledData);
            if (int16Pcm.buffer.byteLength > 0) {
                 console.log(`UI_LOG: onaudioprocess - Sending ${int16Pcm.buffer.byteLength} bytes (16kHz, 16-bit PCM).`);
                 voiceWs.send(int16Pcm.buffer);
            }
        };
        source.connect(audioProcessorNode);
        audioProcessorNode.connect(audioContext.destination); // Necessary for onaudioprocess to fire
        console.log("UI_LOG: Microphone started & connected to processor. Browser's AudioContext sample rate:", audioContext.sampleRate);
        return mediaStream;
    } catch (err) { /* ... (your existing error handling) ... */ 
        console.error("UI_LOG: Error in startMicrophone:", err);
        return null;
    }
}

function stopMicrophone() {
    console.log("UI_LOG: stopMicrophone called.");
    if (mediaStream) { /* ... */ }
    if (audioProcessorNode) { /* ... */ }
    // ... (as before)
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
        console.log("UI_LOG: Microphone stream tracks stopped.");
    }
    if (audioProcessorNode) {
        audioProcessorNode.disconnect(); // Disconnect from source and destination
        audioProcessorNode.onaudioprocess = null; // Remove handler
        audioProcessorNode = null;
        console.log("UI_LOG: AudioProcessorNode disconnected.");
    }
}

function downsampleBuffer(buffer, inputRate, outputRate) { /* ... as before ... */ 
    if (inputRate === outputRate) return buffer;
    const ratio = inputRate / outputRate;
    const newLength = Math.round(buffer.length / ratio);
    const result = new Float32Array(newLength);
    let i = 0, j = 0;
    while (i < newLength) {
        result[i++] = buffer[Math.floor(j)];
        j += ratio;
    }
    return result;
}
function convertFloat32ToInt16(buffer) { /* ... as before ... */ 
    let l = buffer.length; const buf = new Int16Array(l);
    while (l--) { buf[l] = Math.min(1, buffer[l]) * 0x7FFF; }
    return buf;
}


// --- REFINED: Audio Playback with Queuing ---
async function playNextChunkFromQueue() {
    if (isCurrentlyPlayingFromServer || browserAudioPlaybackQueue.length === 0) {
        if (browserAudioPlaybackQueue.length === 0 && !isCurrentlyPlayingFromServer) {
            // console.log("UI_LOG: Playback queue empty and not currently playing.");
        }
        return; // Either already playing, or nothing to play
    }

    isCurrentlyPlayingFromServer = true; // Set flag: we are now starting a playback sequence
    const arrayBuffer = browserAudioPlaybackQueue.shift(); // Get next chunk

    console.log(`UI_LOG: playNextChunkFromQueue - Dequeued chunk. Queue size: ${browserAudioPlaybackQueue.length}. Playing chunk size: ${arrayBuffer.byteLength}`);

    if (!audioContext) {
        try {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            console.log("UI_LOG: AudioContext initialized for playback. State:", audioContext.state);
            if (audioContext.state === 'suspended') {
                console.log("UI_LOG: AudioContext is suspended (playback), attempting to resume...");
                await audioContext.resume();
                console.log("UI_LOG: AudioContext resumed successfully for playback. State:", audioContext.state);
            }
        } catch (e) {
            console.error("UI_LOG: Error initializing or resuming AudioContext:", e);
            addStatusMessage("Audio system error. Please try again.", "error-text");
            isCurrentlyPlayingFromServer = false; // Reset flag
            // Optionally, attempt to play the next chunk if there was an error with the context? Or just stop.
            return;
        }
    }
    // Ensure context is running if already initialized
    if (audioContext.state === 'suspended') {
         console.log("UI_LOG: AudioContext was suspended before playback of this chunk, attempting resume...");
         try {
            await audioContext.resume();
            console.log("UI_LOG: AudioContext resumed. Current state:", audioContext.state);
         } catch(e) {
            console.error("UI_LOG: Failed to resume AudioContext before playing chunk.", e);
            isCurrentlyPlayingFromServer = false;
            // Consider re-adding the chunk to the queue if resume fails
            // browserAudioPlaybackQueue.unshift(arrayBuffer);
            return;
         }
    }


    if (!arrayBuffer || arrayBuffer.byteLength === 0) {
        console.warn("UI_LOG: playNextChunkFromQueue - Attempted to play empty/null chunk.");
        isCurrentlyPlayingFromServer = false; // Reset flag
        playNextChunkFromQueue();       // Immediately try to play the next one if this was invalid
        return;
    }

    try {
        const int16Array = new Int16Array(arrayBuffer);
        const float32Array = new Float32Array(int16Array.length);
        for (let i = 0; i < int16Array.length; i++) {
            float32Array[i] = int16Array[i] / 32768.0; 
        }

        const audioBuffer = audioContext.createBuffer(
            1,                          
            float32Array.length,        
            expectedPlaybackSampleRate  // 24000 for Gemini output
        );
        audioBuffer.getChannelData(0).set(float32Array);

        const source = audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(audioContext.destination);
        
        source.onended = () => {
            console.log("UI_LOG: Playback of an audio chunk (source node) has truly ended.");
            source.disconnect(); 
            isCurrentlyPlayingFromServer = false; // Reset flag *after* this one is done
            playNextChunkFromQueue();       // Now, try to play the next chunk in the queue
        };
        
        console.log("UI_LOG: playNextChunkFromQueue - Calling source.start() for current chunk.");
        source.start(); // Play the current chunk

    } catch (e) {
        console.error("UI_LOG: Error in playNextChunkFromQueue during Web Audio API operations:", e);
        addStatusMessage("Error playing audio response.", "error-text");
        isCurrentlyPlayingFromServer = false; // Reset flag
        // Optionally try to play the next chunk after an error, or clear queue.
        // playNextChunkFromQueue(); 
    }
}

// --- Mic Button Logic ---
async function toggleListening() {
    console.log("UI_LOG: toggleListening called. isListening:", isListening);
    if (!voiceWs || voiceWs.readyState !== WebSocket.OPEN) {
        addStatusMessage("Connecting to voice service...", "connection-open-text");
        if (micButton) micButton.disabled = true;
        try {
            await connectVoiceWebSocket();
            console.log("UI_LOG: Voice WebSocket connection successful (from toggleListening).");
            // handleVoiceWebSocketOpen should have re-enabled micButton if successful
        } catch (err) {
            console.error("UI_LOG: Failed to connect voice WebSocket on toggle:", err);
            if (micButton) micButton.disabled = false;
            addStatusMessage("Failed to connect to voice service.", "error-text");
            return;
        }
    }
    
    if (isListening) {
        stopListening();
    } else {
        // Ensure AudioContext is running before starting microphone
        if (!audioContext || audioContext.state === 'suspended') {
            if(!audioContext) audioContext = new (window.AudioContext || window.webkitAudioContext)();
            try {
                console.log("UI_LOG: Resuming AudioContext before starting microphone.");
                await audioContext.resume();
                console.log("UI_LOG: AudioContext resumed for mic start. State:", audioContext.state);
            } catch (e) {
                console.error("UI_LOG: Could not resume audio context on mic press:", e);
                addStatusMessage("Audio system could not start. Please click again.", "error-text");
                return;
            }
        }

        const stream = await startMicrophone(); // startMicrophone also tries to resume context
        if (stream) {
            isListening = true;
            micIcon.textContent = "mic_off";
            micButton.classList.add("is-listening");
            if (messageInput) messageInput.disabled = true;
            if (sendButton) sendButton.disabled = true;
            showThinkingIndicator(true);
            addStatusMessage("Listening... Speak now.", "connection-open-text");
        } else {
            console.error("UI_LOG: Failed to start microphone in toggleListening.");
            // Potentially reset micButton state if startMicrophone failed
             if (micIcon) micIcon.textContent = "mic";
             if (micButton) micButton.classList.remove("is-listening");
        }
    }
}

function stopListening() {
    console.log("UI_LOG: stopListening called. Current isListening state:", isListening);
    if (isListening) { // Only run if actually listening
        stopMicrophone();
        isListening = false; // Set before UI updates
        if (micIcon) micIcon.textContent = "mic";
        if (micButton) micButton.classList.remove("is-listening");
        if (messageInput) messageInput.disabled = false;
        if (sendButton && textWs && textWs.readyState === WebSocket.OPEN) sendButton.disabled = false;
        hideThinkingIndicator();
        addStatusMessage("Stopped listening.", "connection-closed-text");
    } else { // If called when not listening, just ensure UI is correct
        if (micIcon) micIcon.textContent = "mic";
        if (micButton) micButton.classList.remove("is-listening");
    }
}

// --- Form Submission (Text) ---
function submitMessageHandler(e) { /* ... as before ... */ 
    e.preventDefault();
    if (!textWs || textWs.readyState !== WebSocket.OPEN) { return false; }
    const messageText = messageInput.value.trim();
    if (messageText) {
        addMessageToUI(messageText, "user");
        showThinkingIndicator(false);
        try {
            textWs.send(JSON.stringify({ message: messageText })); // Send as JSON for text chat
            messageInput.value = ""; messageInput.focus();
        } catch (error) { 
            console.error("UI_LOG: Error sending text message:", error);
            hideThinkingIndicator();
            addStatusMessage(`Failed to send: ${error.message}`, "error-text");
        }
    }
    return false;
}
function addSubmitHandler() { /* ... as before ... */ 
    if (messageForm && submitMessageHandler) {
        messageForm.removeEventListener("submit", submitMessageHandler);
        messageForm.addEventListener("submit", submitMessageHandler);
        console.log("UI_LOG: Text submit handler assigned.");
    }
}
function removeSubmitHandler() { /* ... as before ... */ 
    if (messageForm && submitMessageHandler) {
        messageForm.removeEventListener("submit", submitMessageHandler);
        console.log("UI_LOG: Text submit handler removed.");
    }
}

// --- Initialization ---
export function initWebSocketApp() {
  if (appInitialized) { console.warn("UI_LOG: WebSocket app already initialized."); return; }
  console.log("UI_LOG: Initializing WebSocket application logic (Text & Voice)...");
  messageForm = document.getElementById("message-form"); /* ... all element getters ... */
  messageInput = document.getElementById("message");
  messagesDiv = document.getElementById("messages");
  sendButton = document.getElementById("send-button");
  micButton = document.getElementById("mic-button");
  micIcon = document.getElementById("mic-icon");

  if (!messageForm || !messageInput || !messagesDiv || !sendButton || !micButton || !micIcon) { /* ... error ... */ return; }
  console.log("UI_LOG: All essential UI elements found.");
  if (sendButton) sendButton.disabled = true;
  if (micButton) micButton.disabled = true;
  micButton.addEventListener("click", toggleListening);
  console.log("UI_LOG: Mic button event listener added.");
  appInitialized = true;
  connectTextWebSocket(); // Text WS connects on load
  console.log("UI_LOG: WebSocket App module initialized. Text WS connecting. Voice WS on demand.");
}