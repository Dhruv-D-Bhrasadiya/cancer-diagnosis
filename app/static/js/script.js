document.getElementById("prediction-form").addEventListener("submit", async function(e) {
    e.preventDefault();

    const gene = document.getElementById("gene").value;
    const variation = document.getElementById("variation").value;
    const text = document.getElementById("text").value;
    
    const buttonText = document.querySelector(".button-text");
    const loader = document.querySelector(".loader");
    const resultContainer = document.getElementById("result-container");
    const resultElement = document.getElementById("predicted-class");

    // UI Loading State
    buttonText.style.display = "none";
    loader.style.display = "block";
    resultContainer.classList.remove("show");
    
    // Slight delay for smooth animation out
    setTimeout(() => resultContainer.classList.add("hidden"), 300);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ gene, variation, text })
        });

        const data = await response.json();

        // Restore UI
        buttonText.style.display = "block";
        loader.style.display = "none";

        resultContainer.classList.remove("hidden", "error");
        
        if (response.ok) {
            resultElement.innerText = data.mock ? `${data.prediction} (Mock)` : data.prediction;
            // Force reflow for animation
            void resultContainer.offsetWidth;
            resultContainer.classList.add("show");
        } else {
            resultContainer.classList.add("error");
            resultElement.innerText = "Error";
            console.error(data.error);
            void resultContainer.offsetWidth;
            resultContainer.classList.add("show");
        }
    } catch (error) {
        buttonText.style.display = "block";
        loader.style.display = "none";
        
        resultContainer.classList.remove("hidden");
        resultContainer.classList.add("error", "show");
        resultElement.innerText = "Error";
        console.error("Network error:", error);
    }
});
