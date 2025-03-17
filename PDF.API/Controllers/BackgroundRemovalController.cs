using Microsoft.AspNetCore.Mvc;
using PDF.API.Services;

namespace PDF.API.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class BackgroundRemovalController : ControllerBase
    {
        private readonly IBackgroundRemovalService _backgroundRemovalService;
        private readonly ILogger<BackgroundRemovalController> _logger;

        public BackgroundRemovalController(
            IBackgroundRemovalService backgroundRemovalService,
            ILogger<BackgroundRemovalController> logger)
        {
            _backgroundRemovalService = backgroundRemovalService;
            _logger = logger;
        }

        /// <summary>
        /// Removes the background from an uploaded image
        /// </summary>
        /// <returns>The processed image with background removed</returns>
        [HttpPost("remove")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status400BadRequest)]
        [ProducesResponseType(StatusCodes.Status500InternalServerError)]
        public async Task<IActionResult> RemoveBackground(IFormFile image)
        {
            if (image == null || image.Length == 0)
            {
                _logger.LogWarning("No image uploaded.");
                return BadRequest("Please upload an image file.");
            }

            // Validate file type
            var allowedExtensions = new[] { ".jpg", ".jpeg", ".png", ".bmp", ".webp" };
            var extension = Path.GetExtension(image.FileName).ToLowerInvariant();
            if (!allowedExtensions.Contains(extension))
            {
                _logger.LogWarning($"Invalid file type: {extension}");
                return BadRequest($"Invalid file type '{extension}'. Supported formats: JPG, JPEG, PNG, BMP, WEBP.");
            }

            try
            {
                _logger.LogInformation($"Processing image: {image.FileName}, size: {image.Length} bytes");

                // Use the appropriate service method based on your provided examples
                var outputFilePath = await _backgroundRemovalService.RemoveBackgroundAsync(image);

                if (string.IsNullOrEmpty(outputFilePath) || !System.IO.File.Exists(outputFilePath))
                {
                    _logger.LogError("Background removal service returned an invalid output path.");
                    return StatusCode(500, "Background removal failed: output file not found.");
                }

                // Read the file and return it as a download
                var resultBytes = await System.IO.File.ReadAllBytesAsync(outputFilePath);

                _logger.LogInformation("Background removal successful.");
                return File(resultBytes, "image/png", $"background_removed_{Path.GetFileName(image.FileName)}.png");
            }
            catch (ArgumentException ex)
            {
                _logger.LogWarning(ex, "Invalid input parameters.");
                return BadRequest($"Invalid input: {ex.Message}");
            }
            catch (FileNotFoundException ex)
            {
                _logger.LogError(ex, "Required file not found.");
                return StatusCode(500, "Processing error: Required file missing.");
            }
            catch (UnauthorizedAccessException ex)
            {
                _logger.LogError(ex, "Permission issue while processing the image.");
                return StatusCode(500, "Processing error: Insufficient permissions.");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Unexpected error occurred while removing background.");
                return StatusCode(500, "An error occurred while processing the image.");
            }
        }

    }
}