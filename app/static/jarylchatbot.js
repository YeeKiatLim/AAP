const uploadContainer = document.getElementById('upload-container');
        const textContainer = document.getElementById('text-container');
        const toggleUpload = document.getElementById('toggle-upload');
        const toggleText = document.getElementById('toggle-text');
        const fileUpload = document.getElementById('file-upload');
        const textInput = document.getElementById('text-input');
        const submitButton = document.getElementById('submit-button');
        const chatButton = document.getElementById('chat-button');
        const chatIframe = document.getElementById('chat-iframe');

        toggleUpload.addEventListener('click', () => {
            toggleUpload.classList.add('active');
            toggleText.classList.remove('active');
            uploadContainer.classList.remove('hidden');
            textContainer.classList.add('hidden');
        });

        toggleText.addEventListener('click', () => {
            toggleText.classList.add('active');
            toggleUpload.classList.remove('active');
            textContainer.classList.remove('hidden');
            uploadContainer.classList.add('hidden');
        });

        uploadContainer.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadContainer.classList.add('hover');
        });

        uploadContainer.addEventListener('dragleave', () => {
            uploadContainer.classList.remove('hover');
        });

        uploadContainer.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadContainer.classList.remove('hover');

            const files = e.dataTransfer.files;
            if (files.length) {
                fileUpload.files = files;
                uploadFile(files[0]);
            }
        });

        fileUpload.addEventListener('change', () => {
            if (fileUpload.files.length) {
                uploadFile(fileUpload.files[0]);
            }
        });

        function uploadFile(file) {
            const formData = new FormData();
            formData.append('file', file);

            fetch('/classifySMS', {
                method: 'POST',
                body: formData
            })
                .then(response => response.json())
                .then(data => {
                    const result = data.result;
                    if (result === 'spam') {
                        window.location.href = '/static/spam.html';  // Redirect to spam page
                    } else if (result === 'ham') {
                        window.location.href = '/static/ham.html';   // Redirect to ham page
                    } else {
                        alert('Unexpected result: ' + result);
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('An error occurred while processing your request.');
                });
        }

        submitButton.addEventListener('click', () => {
            const text = textInput.value;
            if (text.trim() !== "") {
                const formData = new FormData();
                formData.append('text', text);

                fetch('/classifySMS', {
                    method: 'POST',
                    body: formData
                })
                    .then(response => response.json())
                    .then(data => {
                        const result = data.result;
                        if (result === 'spam') {
                            window.location.href = '/static/spam.html';  // Redirect to spam page
                        } else if (result === 'ham') {
                            window.location.href = '/static/ham.html';   // Redirect to ham page
                        } else {
                            alert('Unexpected result: ' + result);
                        }
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        alert('An error occurred while processing your request.');
                    });
            } else {
                alert('Please enter some text.');
            }
        });

        chatButton.addEventListener('click', () => {
            const icon = chatButton.querySelector('.icon');

            if (chatIframe.classList.contains('open')) {
                chatIframe.classList.remove('open');
                chatButton.classList.remove('active'); // Remove active class

                icon.classList.add('spin');
                setTimeout(() => {
                    icon.classList.remove('fa-times');
                    icon.classList.add('fa-comments');
                    icon.classList.remove('spin');
                }, 250); // Half of the animation duration for smooth transition
            } else {
                chatIframe.classList.add('open');
                chatButton.classList.add('active'); // Add active class

                icon.classList.add('spin');
                setTimeout(() => {
                    icon.classList.remove('fa-comments');
                    icon.classList.add('fa-times');
                    icon.classList.remove('spin');
                }, 250); // Half of the animation duration for smooth transition
            }
        });

        // Adjust textarea height based on input
        textInput.addEventListener('input', function () {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });