const demoapp = {
    data() {
        return {
            recorder: null,
            audio_file: null
        }
    },
    methods: {
        uploadFile() {
            this.audio_file = this.$refs.file.files[0]
            this.playAudio()
        },
        playAudio: function () {
            let audio = document.getElementById('audio-preview')
            let reader = new FileReader()

            reader.readAsDataURL(this.audio_file)
            reader.addEventListener('load', function () {
                audio.src = reader.result
            })
        },
        transcribeAudio: function () {
            const formData = new FormData()
            formData.append('audio_file', this.audio_file)
            const headers = { 'Content-Type': 'multipart/form-data' }
            axios.post('http://127.0.0.1:8888/transcribe/', formData, { headers })
                .then((res) => {
                    document.getElementById('output').innerHTML = "Text: " + res.data.text[0]
                })
            /*fetch("http://127.0.0.1:8888/transcribe/", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(("#myForm").serializeArray())
            })
            .then(response => response.json())
            .then(result => {
                document.getElementById('output').innerHTML = "Text: " + result.text
            })*/
        }
    }
}

Vue.createApp(demoapp).mount("#demoapp")