document.addEventListener('DOMContentLoaded', function () {
    const demoApp = {
        audio_file: null,

        init: function () {
            document.getElementById('file-input').addEventListener('change', this.uploadFile.bind(this));
            document.getElementById('transcribe-btn').addEventListener('click', this.transcribeAudio.bind(this));
            document.getElementById('record-btn').addEventListener('click', this.toggleRecording.bind(this));
        },

        uploadFile: function (event) {
            this.audio_file = event.target.files[0];
            document.getElementById('audio-preview').style.display = 'block';
            document.getElementById('audio-preview').src = this.audio_file;
        },

        toggleRecording: function () {
            if (this.isRecording) {
                this.stopRecording();
            } else {
                this.startRecording();
            }
        },

        startRecording: function () {
            navigator.mediaDevices.getUserMedia({ audio: true })
                .then(stream => {
                    this.mediaRecorder = new MediaRecorder(stream);
                    this.mediaRecorder.start();
                    this.isRecording = true;
                    document.getElementById('record-btn').innerText = "Stop Recording";

                    const audioChunks = [];
                    this.mediaRecorder.addEventListener("dataavailable", event => {
                        audioChunks.push(event.data);
                    });

                    this.mediaRecorder.addEventListener("stop", () => {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                        this.audio_file = audioBlob;

                        const audioUrl = URL.createObjectURL(audioBlob);
                        document.getElementById('audio-preview').style.display = 'block';
                        document.getElementById('audio-preview').src = audioUrl;

                        this.isRecording = false;
                        document.getElementById('record-btn').innerText = "Record Audio";
                    });
                })
                .catch(error => {
                    console.error('Error accessing microphone:', error);
                });
        },

        stopRecording: function () {
            if (this.mediaRecorder && this.mediaRecorder.state === "recording") {
                this.mediaRecorder.stop();
            }
        },

        transcribeAudio: function () {
            const formData = new FormData();
            console.log(this.audio_file);
            formData.append('audio_file', this.audio_file);
            var text = "";
            console.log(formData);

            fetch('http://127.0.0.1:8888/transcribe', {
                method: 'POST',
                body: formData
            })
                .then(response => response.json())
                .then(result => {
                    console.log(result);
                    text = result.text[0];
                    document.getElementById('output').innerText = "Text: " + result.text[0];
                    document.getElementById('output').style.display = 'block';
                    this.predictSpam(text);
                    document.getElementById('scam_prob').style.display = 'block';
                    document.getElementById('legit_prob').style.display = 'block';
                    document.getElementById('pred').style.display = 'block';
                })
                .catch(error => console.error('Error:', error));
        },

        predictSpam: function (text) {

            fetch('http://127.0.0.1:8888/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: text })
            })
                .then(response => response.json())
                .then(result => {
                    console.log(result);
                    document.getElementById('scam_prob').innerText = "Scam Probability: " + result["Probability of Scam"].toFixed(2) + "%";
                    document.getElementById('legit_prob').innerText = "Legitimacy Probability: " + result["Probability of Legitimacy"].toFixed(2) + "%";
                    document.getElementById('pred').innerText = "Prediction: " + result.prediction;
                })
                .catch(error => console.error('Error:', error));
        },
    };

    demoApp.init();
});