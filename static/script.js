// Check if the browser supports Speech Recognition
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
if (!SpeechRecognition) {
    alert("Speech Recognition is not supported in this browser. Please use Google Chrome.");
} else {
    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true; // Show real-time speech conversion
    recognition.lang = 'en-US'; // Set language


    let temp = 0;
    let micTimeout;

    // Start Button
    document.getElementById("mic-icon").addEventListener("click", () => {

        let mic = document.getElementById('mic-icon');

        if (temp == 0) {
            recognition.start();
            mic.src = "/static/pouse.png";
            temp = 1;
            console.log(temp);
            micTimeout = setTimeout(() => {
                recognition.stop();
                mic.src = "/static/microphone.png";  // Reset mic icon
                temp = 0;
            }, 10000); // Stops after 10000ms (10 seconds)
        }
        else {
            recognition.stop();
            mic.src = "/static/microphone.png";
            temp = 0;
            console.log(temp);
        }
    });

    // Capture Speech and Display Text
    recognition.onresult = (event) => {
        let transcript = "";
        for (let i = event.resultIndex; i < event.results.length; i++) {
            transcript += event.results[i][0].transcript;
            console.log(transcript);
        }
        document.getElementById("input_message").value = transcript;
    };

    // Handle errors
    recognition.onerror = (event) => {
        alert("Error: " + event.error);
    };
}