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

            fetch('http://127.0.0.1:8888/transcribe/', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(result => {
                document.getElementById('output').innerText = "Text: " + result.text[0];
                document.getElementById('output').style.display = 'block';
            })
            .catch(error => console.error('Error:', error));
        }
    };

    demoApp.init();
});