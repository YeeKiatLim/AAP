document.addEventListener('DOMContentLoaded', function () {
    const demoApp = {
        recorder: null,
        audio_file: null,

        init: function () {
            document.getElementById('file-input').addEventListener('change', this.uploadFile.bind(this));
            document.getElementById('transcribe-btn').addEventListener('click', this.transcribeAudio.bind(this));
        },

        uploadFile: function (event) {
            this.audio_file = event.target.files[0];
            document.getElementById('audio-preview').style.display = 'block';
            this.playAudio();            
        },

        playAudio: function () {
            const audio = document.getElementById('audio-preview');
            const reader = new FileReader();

            reader.readAsDataURL(this.audio_file);
            reader.addEventListener('load', function () {
                audio.src = reader.result;
            });
        },

        transcribeAudio: function () {
            const formData = new FormData();
            formData.append('audio_file', this.audio_file);
            var text = "";

            fetch('http://127.0.0.1:8888/transcribe', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(result => {
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