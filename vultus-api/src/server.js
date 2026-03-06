import { createApp } from './app.js';
import { config } from './config.js';
import { logger } from './logger.js';

const app = createApp();

logger.info(`[VULTUS-API] Starting server in ${config.env} mode...`);

app.listen(config.port, () => {
  logger.info(`[VULTUS-API] Server is up and running on port ${config.port}`);
  logger.info(`[VULTUS-API] Configured Database: ${config.databaseUrl.split('@')[1] || 'Embedded'}`);
  logger.info(`[VULTUS-API] ML Inference URL: ${config.faceInferenceUrl}`);
});
