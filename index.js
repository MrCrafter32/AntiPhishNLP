const express = require('express');
const { pipeline } = require('@xenova/transformers');

const app = express();
app.use(express.json());

let classifier;

async function initializeModel() {
    classifier = await pipeline('text-classification', 'mrcrafter32/AntiPhishX-BERT');
}

initializeModel();

app.post('/predict', async (req, res) => {
    const email_text = req.body.email_text;

    if (!email_text) {
        return res.status(400).json({ error: 'No email text provided' });
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