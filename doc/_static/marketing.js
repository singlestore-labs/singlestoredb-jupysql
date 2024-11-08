document.addEventListener("DOMContentLoaded", (event) => {
    options = [
        {
            text: "Deploy Streamlit apps for free on ",
            link: "Ploomber Cloud!",
            url: "https://platform.ploomber.io/register/?utm_medium=readthedocs&utm_source=jupysql",
        },
        {
            text: "Deploy Shiny apps for free on ",
            link: "Ploomber Cloud!",
            url: "https://platform.ploomber.io/register/?utm_medium=readthedocs&utm_source=jupysql",
        },
        {
            text: "Deploy Dash apps for free on ",
            link: "Ploomber Cloud!",
            url: "https://platform.ploomber.io/register/?utm_medium=readthedocs&utm_source=jupysql",
        },
        {
            text: "Try our new Streamlit ",
            link: "AI Editor!",
            url: "https://editor.ploomber.io/?utm_medium=readthedocs&utm_source=jupysql",
        }
    ]
    const announcementElement = document.querySelector("#ploomber-top-announcement");
    if (announcementElement) {
        const updateAnnouncement = (firstTime = false) => {
            let randomIndex;
            let currentContent = announcementElement.textContent;

            // Keep selecting a new random index until we get a different announcement
            do {
                randomIndex = Math.floor(Math.random() * options.length);
            } while (options[randomIndex].text + options[randomIndex].link === currentContent);

            const option = options[randomIndex];

            if (firstTime) {
                // First time - just set content without transition
                announcementElement.innerHTML = `${option.text}<a href="${option.url}" target="_blank" rel="noopener noreferrer">${option.link}</a>`;
            } else {
                // Subsequent updates - use transition
                announcementElement.style.opacity = 0;
                announcementElement.style.transition = 'opacity 0.5s ease';

                setTimeout(() => {
                    announcementElement.innerHTML = `${option.text}<a href="${option.url}" target="_blank" rel="noopener noreferrer">${option.link}</a>`;
                    announcementElement.style.opacity = 1;
                }, 500);
            }
        };

        // Set initial content without transition
        updateAnnouncement(true);

        // Update every 5 seconds with transition
        setInterval(() => updateAnnouncement(false), 5000);
    }
});
