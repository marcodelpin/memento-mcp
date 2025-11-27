import axios from 'axios';
import { EmbeddingService, type EmbeddingModelInfo } from './EmbeddingService.js';
import { logger } from '../utils/logger.js';

/**
 * Configuration for Ollama embedding service
 */
export interface OllamaEmbeddingConfig {
  /**
   * Ollama server URL (default: http://localhost:11434)
   */
  baseUrl?: string;

  /**
   * Model name to use for embeddings (default: nomic-embed-text)
   */
  model?: string;

  /**
   * Embedding dimensions (auto-detected from first call if not specified)
   */
  dimensions?: number;

  /**
   * Request timeout in milliseconds (default: 30000)
   */
  timeout?: number;
}

/**
 * Ollama API response structure for embeddings
 */
interface OllamaEmbeddingResponse {
  embedding: number[];
}

/**
 * Service implementation that generates embeddings using Ollama's API
 */
export class OllamaEmbeddingService extends EmbeddingService {
  private baseUrl: string;
  private model: string;
  private dimensions: number;
  private timeout: number;
  private dimensionsDetected: boolean = false;

  /**
   * Create a new Ollama embedding service
   *
   * @param config - Configuration for the service
   */
  constructor(config: OllamaEmbeddingConfig = {}) {
    super();

    this.baseUrl = config.baseUrl || process.env.OLLAMA_BASE_URL || 'http://localhost:11434';
    this.model = config.model || process.env.OLLAMA_EMBEDDING_MODEL || 'nomic-embed-text';
    this.dimensions = config.dimensions || 768; // nomic-embed-text default
    this.timeout = config.timeout || 30000;

    // Remove trailing slash from baseUrl
    if (this.baseUrl.endsWith('/')) {
      this.baseUrl = this.baseUrl.slice(0, -1);
    }

    logger.debug('OllamaEmbeddingService initialized', {
      baseUrl: this.baseUrl,
      model: this.model,
      dimensions: this.dimensions,
    });
  }

  /**
   * Generate an embedding for a single text
   *
   * @param text - Text to generate embedding for
   * @returns Promise resolving to embedding vector
   */
  override async generateEmbedding(text: string): Promise<number[]> {
    const endpoint = `${this.baseUrl}/api/embeddings`;

    logger.debug('Generating Ollama embedding', {
      text: text.substring(0, 50) + '...',
      model: this.model,
      endpoint,
    });

    try {
      const response = await axios.post<OllamaEmbeddingResponse>(
        endpoint,
        {
          model: this.model,
          prompt: text,
        },
        {
          headers: {
            'Content-Type': 'application/json',
          },
          timeout: this.timeout,
        }
      );

      if (!response.data || !response.data.embedding) {
        logger.error('Invalid response from Ollama API', { response: response.data });
        throw new Error('Invalid response from Ollama API - missing embedding data');
      }

      const embedding = response.data.embedding;

      if (!Array.isArray(embedding) || embedding.length === 0) {
        logger.error('Invalid embedding returned', { embedding });
        throw new Error('Invalid embedding returned from Ollama API');
      }

      // Auto-detect dimensions on first successful call
      if (!this.dimensionsDetected) {
        this.dimensions = embedding.length;
        this.dimensionsDetected = true;
        logger.info('Ollama embedding dimensions detected', { dimensions: this.dimensions });
      }

      logger.debug('Generated Ollama embedding', {
        length: embedding.length,
        sample: embedding.slice(0, 5),
      });

      // Normalize the embedding vector
      this._normalizeVector(embedding);

      return embedding;
    } catch (error: unknown) {
      const axiosError = error as {
        isAxiosError?: boolean;
        response?: {
          status?: number;
          data?: unknown;
        };
        code?: string;
        message?: string;
      };

      if (axiosError.isAxiosError) {
        const statusCode = axiosError.response?.status;
        const responseData = axiosError.response?.data;

        logger.error('Ollama API error', {
          status: statusCode,
          data: responseData,
          message: axiosError.message,
          code: axiosError.code,
        });

        if (axiosError.code === 'ECONNREFUSED') {
          throw new Error(`Ollama server not reachable at ${this.baseUrl}`);
        }

        if (statusCode === 404) {
          throw new Error(`Ollama model "${this.model}" not found - try 'ollama pull ${this.model}'`);
        }

        throw new Error(`Ollama API error (${statusCode || axiosError.code || 'unknown'})`);
      }

      const errorMessage = this._getErrorMessage(error);
      logger.error('Failed to generate Ollama embedding', { error: errorMessage });
      throw new Error(`Error generating Ollama embedding: ${errorMessage}`);
    }
  }

  /**
   * Generate embeddings for multiple texts
   *
   * @param texts - Array of texts to generate embeddings for
   * @returns Promise resolving to array of embedding vectors
   */
  override async generateEmbeddings(texts: string[]): Promise<number[][]> {
    // Ollama doesn't support batch embeddings, so we process sequentially
    const embeddings: number[][] = [];

    for (const text of texts) {
      const embedding = await this.generateEmbedding(text);
      embeddings.push(embedding);
    }

    return embeddings;
  }

  /**
   * Get information about the embedding model
   *
   * @returns Model information
   */
  override getModelInfo(): EmbeddingModelInfo {
    return {
      name: this.model,
      dimensions: this.dimensions,
      version: '1.0.0',
    };
  }

  /**
   * Extract error message from error object
   *
   * @private
   * @param error - Error object
   * @returns Error message string
   */
  private _getErrorMessage(error: unknown): string {
    if (error instanceof Error) {
      return error.message;
    }
    return String(error);
  }

  /**
   * Normalize a vector to unit length (L2 norm)
   *
   * @private
   * @param vector - Vector to normalize in-place
   */
  private _normalizeVector(vector: number[]): void {
    let magnitude = 0;
    for (let i = 0; i < vector.length; i++) {
      magnitude += vector[i] * vector[i];
    }
    magnitude = Math.sqrt(magnitude);

    if (magnitude > 0) {
      for (let i = 0; i < vector.length; i++) {
        vector[i] /= magnitude;
      }
    } else {
      vector[0] = 1;
    }
  }
}
