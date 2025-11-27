import type { EmbeddingService } from './EmbeddingService.js';
import { DefaultEmbeddingService } from './DefaultEmbeddingService.js';
import { OpenAIEmbeddingService } from './OpenAIEmbeddingService.js';
import { OllamaEmbeddingService } from './OllamaEmbeddingService.js';
import { OpenRouterEmbeddingService } from './OpenRouterEmbeddingService.js';
import { logger } from '../utils/logger.js';

/**
 * Configuration options for embedding services
 */
export interface EmbeddingServiceConfig {
  provider?: string;
  model?: string;
  dimensions?: number;
  apiKey?: string;
  baseUrl?: string;
  timeout?: number;
  siteUrl?: string;
  siteName?: string;
  [key: string]: unknown;
}

/**
 * Type definition for embedding service provider creation function
 */
type EmbeddingServiceProvider = (config?: EmbeddingServiceConfig) => EmbeddingService;

/**
 * Factory for creating embedding services
 */
export class EmbeddingServiceFactory {
  /**
   * Registry of embedding service providers
   */
  private static providers: Record<string, EmbeddingServiceProvider> = {};

  /**
   * Register a new embedding service provider
   *
   * @param name - Provider name
   * @param provider - Provider factory function
   */
  static registerProvider(name: string, provider: EmbeddingServiceProvider): void {
    EmbeddingServiceFactory.providers[name.toLowerCase()] = provider;
  }

  /**
   * Reset the provider registry - used primarily for testing
   */
  static resetRegistry(): void {
    EmbeddingServiceFactory.providers = {};
  }

  /**
   * Get a list of available provider names
   *
   * @returns Array of provider names
   */
  static getAvailableProviders(): string[] {
    return Object.keys(EmbeddingServiceFactory.providers);
  }

  /**
   * Create a service using a registered provider
   *
   * @param config - Configuration options including provider name and service-specific settings
   * @returns The created embedding service
   * @throws Error if the provider is not registered
   */
  static createService(config: EmbeddingServiceConfig = {}): EmbeddingService {
    const providerName = (config.provider || 'default').toLowerCase();
    logger.debug(`EmbeddingServiceFactory: Creating service with provider "${providerName}"`);

    const providerFn = EmbeddingServiceFactory.providers[providerName];

    if (providerFn) {
      try {
        const service = providerFn(config);
        logger.debug(
          `EmbeddingServiceFactory: Service created successfully with provider "${providerName}"`,
          {
            modelInfo: service.getModelInfo(),
          }
        );
        return service;
      } catch (error) {
        logger.error(
          `EmbeddingServiceFactory: Failed to create service with provider "${providerName}"`,
          error
        );
        throw error;
      }
    }

    // If provider not found, throw an error
    logger.error(`EmbeddingServiceFactory: Provider "${providerName}" is not registered`);
    throw new Error(`Provider "${providerName}" is not registered`);
  }

  /**
   * Create an embedding service from environment variables
   *
   * Priority order:
   * 1. MOCK_EMBEDDINGS=true -> DefaultEmbeddingService
   * 2. EMBEDDING_PROVIDER env var -> specified provider
   * 3. OLLAMA_BASE_URL -> OllamaEmbeddingService
   * 4. OPENROUTER_API_KEY -> OpenRouterEmbeddingService
   * 5. OPENAI_API_KEY -> OpenAIEmbeddingService
   * 6. Default -> DefaultEmbeddingService
   *
   * @returns An embedding service implementation
   */
  static createFromEnvironment(): EmbeddingService {
    // Check if we should use mock embeddings (for testing)
    const useMockEmbeddings = process.env.MOCK_EMBEDDINGS === 'true';

    logger.debug('EmbeddingServiceFactory: Creating service from environment variables', {
      mockEmbeddings: useMockEmbeddings,
      embeddingProvider: process.env.EMBEDDING_PROVIDER || 'auto',
      ollamaUrlPresent: !!process.env.OLLAMA_BASE_URL,
      openrouterKeyPresent: !!process.env.OPENROUTER_API_KEY,
      openaiKeyPresent: !!process.env.OPENAI_API_KEY,
    });

    if (useMockEmbeddings) {
      logger.info('EmbeddingServiceFactory: Using mock embeddings for testing');
      return new DefaultEmbeddingService();
    }

    // Check for explicit provider selection
    const explicitProvider = process.env.EMBEDDING_PROVIDER?.toLowerCase();
    if (explicitProvider) {
      return EmbeddingServiceFactory._createExplicitProvider(explicitProvider);
    }

    // Auto-detect based on available environment variables

    // 1. Check for Ollama (local/self-hosted, no API key needed)
    const ollamaBaseUrl = process.env.OLLAMA_BASE_URL;
    if (ollamaBaseUrl) {
      try {
        logger.debug('EmbeddingServiceFactory: Creating Ollama embedding service', {
          baseUrl: ollamaBaseUrl,
          model: process.env.OLLAMA_EMBEDDING_MODEL || 'nomic-embed-text',
        });
        const service = new OllamaEmbeddingService({
          baseUrl: ollamaBaseUrl,
          model: process.env.OLLAMA_EMBEDDING_MODEL,
        });
        logger.info('EmbeddingServiceFactory: Ollama embedding service created successfully', {
          model: service.getModelInfo().name,
          dimensions: service.getModelInfo().dimensions,
        });
        return service;
      } catch (error) {
        logger.error('EmbeddingServiceFactory: Failed to create Ollama service', error);
        // Continue to next provider
      }
    }

    // 2. Check for OpenRouter
    const openrouterApiKey = process.env.OPENROUTER_API_KEY;
    if (openrouterApiKey) {
      try {
        logger.debug('EmbeddingServiceFactory: Creating OpenRouter embedding service', {
          model: process.env.OPENROUTER_EMBEDDING_MODEL || 'openai/text-embedding-3-small',
        });
        const service = new OpenRouterEmbeddingService({
          apiKey: openrouterApiKey,
          model: process.env.OPENROUTER_EMBEDDING_MODEL,
        });
        logger.info('EmbeddingServiceFactory: OpenRouter embedding service created successfully', {
          model: service.getModelInfo().name,
          dimensions: service.getModelInfo().dimensions,
        });
        return service;
      } catch (error) {
        logger.error('EmbeddingServiceFactory: Failed to create OpenRouter service', error);
        // Continue to next provider
      }
    }

    // 3. Check for OpenAI
    const openaiApiKey = process.env.OPENAI_API_KEY;
    const embeddingModel = process.env.OPENAI_EMBEDDING_MODEL || 'text-embedding-3-small';

    if (openaiApiKey) {
      try {
        logger.debug('EmbeddingServiceFactory: Creating OpenAI embedding service', {
          model: embeddingModel,
        });
        const service = new OpenAIEmbeddingService({
          apiKey: openaiApiKey,
          model: embeddingModel,
        });
        logger.info('EmbeddingServiceFactory: OpenAI embedding service created successfully', {
          model: service.getModelInfo().name,
          dimensions: service.getModelInfo().dimensions,
        });
        return service;
      } catch (error) {
        logger.error('EmbeddingServiceFactory: Failed to create OpenAI service', error);
        logger.info('EmbeddingServiceFactory: Falling back to default embedding service');
        // Fallback to default if OpenAI service creation fails
        return new DefaultEmbeddingService();
      }
    }

    // No provider configured, using default embedding service
    logger.info(
      'EmbeddingServiceFactory: No embedding provider configured, using default embedding service'
    );
    return new DefaultEmbeddingService();
  }

  /**
   * Create a service based on explicit provider name
   *
   * @private
   * @param provider - Provider name
   * @returns Embedding service
   */
  private static _createExplicitProvider(provider: string): EmbeddingService {
    switch (provider) {
      case 'ollama':
        return new OllamaEmbeddingService({
          baseUrl: process.env.OLLAMA_BASE_URL,
          model: process.env.OLLAMA_EMBEDDING_MODEL,
        });

      case 'openrouter':
        return new OpenRouterEmbeddingService({
          apiKey: process.env.OPENROUTER_API_KEY,
          model: process.env.OPENROUTER_EMBEDDING_MODEL,
        });

      case 'openai':
        return new OpenAIEmbeddingService({
          apiKey: process.env.OPENAI_API_KEY || '',
          model: process.env.OPENAI_EMBEDDING_MODEL,
        });

      case 'default':
      case 'mock':
        return new DefaultEmbeddingService();

      default:
        logger.warn(`EmbeddingServiceFactory: Unknown provider "${provider}", using default`);
        return new DefaultEmbeddingService();
    }
  }

  /**
   * Create an OpenAI embedding service
   *
   * @param apiKey - OpenAI API key
   * @param model - Optional model name
   * @param dimensions - Optional embedding dimensions
   * @returns OpenAI embedding service
   */
  static createOpenAIService(
    apiKey: string,
    model?: string,
    dimensions?: number
  ): EmbeddingService {
    return new OpenAIEmbeddingService({
      apiKey,
      model,
      dimensions,
    });
  }

  /**
   * Create a default embedding service that generates random vectors
   *
   * @param dimensions - Optional embedding dimensions
   * @returns Default embedding service
   */
  static createDefaultService(dimensions?: number): EmbeddingService {
    return new DefaultEmbeddingService(dimensions);
  }
}

// Register built-in providers
EmbeddingServiceFactory.registerProvider('default', (config = {}) => {
  return new DefaultEmbeddingService(config.dimensions);
});

EmbeddingServiceFactory.registerProvider('openai', (config = {}) => {
  if (!config.apiKey) {
    throw new Error('API key is required for OpenAI embedding service');
  }

  return new OpenAIEmbeddingService({
    apiKey: config.apiKey,
    model: config.model,
    dimensions: config.dimensions,
  });
});

EmbeddingServiceFactory.registerProvider('ollama', (config = {}) => {
  return new OllamaEmbeddingService({
    baseUrl: config.baseUrl,
    model: config.model,
    dimensions: config.dimensions,
    timeout: config.timeout,
  });
});

EmbeddingServiceFactory.registerProvider('openrouter', (config = {}) => {
  if (!config.apiKey) {
    throw new Error('API key is required for OpenRouter embedding service');
  }

  return new OpenRouterEmbeddingService({
    apiKey: config.apiKey,
    model: config.model,
    dimensions: config.dimensions,
    timeout: config.timeout,
    siteUrl: config.siteUrl,
    siteName: config.siteName,
  });
});
