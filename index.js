const express = require('express');

const app = express();
app.use(express.json());

let classifier;

async function initializeModel() {
    try {
        console.log("Starting model initialization...");
        const { pipeline } = await import('@xenova/transformers');
        console.log("Transformers library imported.");
        classifier = await pipeline('text-classification', 'mrcrafter32/AntiPhishX-BERT', {
            progress_callback: (details) => {
                console.log(`${details.progress * 100}% ${details.status}`);
            }
        });
        console.log("Model initialized successfully.");
    } catch (error) {
        console.error('Failed to initialize model:', error);
        if (error.cause && error.cause.message) {
            console.error('Model download error:', error.cause.message); // Log the underlying error
        }
    }
}

initializeModel();

app.post('/predict', async (req, res) => {
    const email_text = req.body.email_text;

    if (!email_text) {
        return res.status(400).json({ error: 'No email text provided' });
    }

    if (!classifier) {
        return res.status(500).json({ error: 'Model not initialized' });
    }

    try {
        const result = await classifier(email_text);
        const label = result[0].label;
        const confidence = result[0].score;

        res.json({
            prediction: label,
            confidence: confidence,
        });
    } catch (error) {
        console.error('Prediction error:', error);
        res.status(500).json({ error: 'Prediction failed' });
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});