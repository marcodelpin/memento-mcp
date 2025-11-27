import axios from 'axios';
import { EmbeddingService, type EmbeddingModelInfo } from './EmbeddingService.js';
import { logger } from '../utils/logger.js';

/**
 * Configuration for OpenRouter embedding service
 */
export interface OpenRouterEmbeddingConfig {
  /**
   * OpenRouter API key
   */
  apiKey?: string;

  /**
   * Model name to use for embeddings (default: openai/text-embedding-3-small)
   */
  model?: string;

  /**
   * Embedding dimensions (model-dependent)
   */
  dimensions?: number;

  /**
   * Request timeout in milliseconds (default: 30000)
   */
  timeout?: number;

  /**
   * Optional site URL for OpenRouter rankings
   */
  siteUrl?: string;

  /**
   * Optional site name for OpenRouter rankings
   */
  siteName?: string;
}

/**
 * OpenRouter API response structure for embeddings
 */
interface OpenRouterEmbeddingResponse {
  data: Array<{
    embedding: number[];
    index: number;
    object: string;
  }>;
  model: string;
  object: string;
  usage: {
    prompt_tokens: number;
    total_tokens: number;
  };
}

/**
 * Service implementation that generates embeddings using OpenRouter's API
 * OpenRouter provides access to multiple embedding models from different providers
 */
export class OpenRouterEmbeddingService extends EmbeddingService {
  private apiKey: string;
  private model: string;
  private dimensions: number;
  private timeout: number;
  private siteUrl: string;
  private siteName: string;
  private apiEndpoint: string;

  /**
   * Create a new OpenRouter embedding service
   *
   * @param config - Configuration for the service
   */
  constructor(config: OpenRouterEmbeddingConfig = {}) {
    super();

    this.apiKey = config.apiKey || process.env.OPENROUTER_API_KEY || '';
    this.model = config.model || process.env.OPENROUTER_EMBEDDING_MODEL || 'openai/text-embedding-3-small';
    this.dimensions = config.dimensions || this._getDefaultDimensions(this.model);
    this.timeout = config.timeout || 30000;
    this.siteUrl = config.siteUrl || process.env.OPENROUTER_SITE_URL || '';
    this.siteName = config.siteName || process.env.OPENROUTER_SITE_NAME || 'memento-mcp';
    this.apiEndpoint = 'https://openrouter.ai/api/v1/embeddings';

    if (!this.apiKey) {
      throw new Error('OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable.');
    }

    logger.debug('OpenRouterEmbeddingService initialized', {
      model: this.model,
      dimensions: this.dimensions,
      siteName: this.siteName,
    });
  }

  /**
   * Get default dimensions for known models
   *
   * @private
   * @param model - Model name
   * @returns Default dimensions for the model
   */
  private _getDefaultDimensions(model: string): number {
    const dimensionMap: Record<string, number> = {
      'openai/text-embedding-3-small': 1536,
      'openai/text-embedding-3-large': 3072,
      'openai/text-embedding-ada-002': 1536,
      'cohere/embed-english-v3.0': 1024,
      'cohere/embed-multilingual-v3.0': 1024,
      'voyage/voyage-2': 1024,
      'voyage/voyage-large-2': 1536,
    };

    return dimensionMap[model] || 1536;
  }

  /**
   * Generate an embedding for a single text
   *
   * @param text - Text to generate embedding for
   * @returns Promise resolving to embedding vector
   */
  override async generateEmbedding(text: string): Promise<number[]> {
    logger.debug('Generating OpenRouter embedding', {
      text: text.substring(0, 50) + '...',
      model: this.model,
    });

    try {
      const headers: Record<string, string> = {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json',
      };

      // Add optional OpenRouter headers
      if (this.siteUrl) {
        headers['HTTP-Referer'] = this.siteUrl;
      }
      if (this.siteName) {
        headers['X-Title'] = this.siteName;
      }

      const response = await axios.post<OpenRouterEmbeddingResponse>(
        this.apiEndpoint,
        {
          input: text,
          model: this.model,
        },
        {
          headers,
          timeout: this.timeout,
        }
      );

      if (!response.data || !response.data.data || !response.data.data[0]) {
        logger.error('Invalid response from OpenRouter API', { response: response.data });
        throw new Error('Invalid response from OpenRouter API - missing embedding data');
      }

      const embedding = response.data.data[0].embedding;

      if (!Array.isArray(embedding) || embedding.length === 0) {
        logger.error('Invalid embedding returned', { embedding });
        throw new Error('Invalid embedding returned from OpenRouter API');
      }

      // Update dimensions if different from default
      if (embedding.length !== this.dimensions) {
        logger.info('Updating embedding dimensions', {
          previous: this.dimensions,
          actual: embedding.length,
        });
        this.dimensions = embedding.length;
      }

      logger.debug('Generated OpenRouter embedding', {
        length: embedding.length,
        sample: embedding.slice(0, 5),
        usage: response.data.usage,
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
        message?: string;
      };

      if (axiosError.isAxiosError) {
        const statusCode = axiosError.response?.status;
        const responseData = axiosError.response?.data;

        logger.error('OpenRouter API error', {
          status: statusCode,
          data: responseData,
          message: axiosError.message,
        });

        if (statusCode === 401) {
          throw new Error('OpenRouter API authentication failed - invalid API key');
        } else if (statusCode === 402) {
          throw new Error('OpenRouter API - insufficient credits');
        } else if (statusCode === 429) {
          throw new Error('OpenRouter API rate limit exceeded - try again later');
        } else if (statusCode === 404) {
          throw new Error(`OpenRouter model "${this.model}" not found or not available for embeddings`);
        }

        throw new Error(`OpenRouter API error (${statusCode || 'unknown'})`);
      }

      const errorMessage = this._getErrorMessage(error);
      logger.error('Failed to generate OpenRouter embedding', { error: errorMessage });
      throw new Error(`Error generating OpenRouter embedding: ${errorMessage}`);
    }
  }

  /**
   * Generate embeddings for multiple texts
   *
   * @param texts - Array of texts to generate embeddings for
   * @returns Promise resolving to array of embedding vectors
   */
  override async generateEmbeddings(texts: string[]): Promise<number[][]> {
    try {
      const headers: Record<string, string> = {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json',
      };

      if (this.siteUrl) {
        headers['HTTP-Referer'] = this.siteUrl;
      }
      if (this.siteName) {
        headers['X-Title'] = this.siteName;
      }

      const response = await axios.post<OpenRouterEmbeddingResponse>(
        this.apiEndpoint,
        {
          input: texts,
          model: this.model,
        },
        {
          headers,
          timeout: this.timeout,
        }
      );

      const embeddings = response.data.data.map((item) => item.embedding);

      // Normalize each embedding vector
      embeddings.forEach((embedding) => {
        this._normalizeVector(embedding);
      });

      return embeddings;
    } catch (error: unknown) {
      const errorMessage = this._getErrorMessage(error);
      throw new Error(`Failed to generate OpenRouter embeddings: ${errorMessage}`);
    }
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
