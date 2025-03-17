using Microsoft.Extensions.Options;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Configuration;
using System;
using System.IO;
using PDF.API.settings;
using System.Diagnostics;


namespace PDF.API.Services
{
    public class BackgroundRemovalService : IBackgroundRemovalService
    {
        private readonly ILogger<BackgroundRemovalService> _logger;
        private readonly BackgroundRemovalSettings _settings;
        private readonly string _tempDirectory;
        private readonly string _pythonPath;
        private readonly string _outputFolderPath;
        private readonly string _scriptPath;

        public BackgroundRemovalService(IConfiguration configuration, ILogger<BackgroundRemovalService> logger, IOptions<BackgroundRemovalSettings> options)
        {
            _logger = logger;
            _settings = options.Value;

            // Read from appsettings.json
            _outputFolderPath = configuration["BackgroundRemoval:OutputFolderPath"];

            _pythonPath = _settings.PythonPath;
            _scriptPath = _settings.ScriptPath;

            // Ensure script path is absolute
            if (!Path.IsPathRooted(_scriptPath))
            {
                _scriptPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, _scriptPath);
            }

            _tempDirectory = _settings.TempDirectory;
            EnsureTempDirectoryExists();

            _logger.LogInformation("BackgroundRemovalService initialized with PythonPath: {PythonPath}, ScriptPath: {ScriptPath}",
                _pythonPath, _scriptPath);
        }

       
       
        public async Task<string> RemoveBackgroundAsync(IFormFile file)
        {
            if (file == null || file.Length == 0)
                throw new ArgumentException("Invalid file uploaded.");

            string uniqueFileName = Guid.NewGuid().ToString() + Path.GetExtension(file.FileName);
            string inputDir = Path.Combine(_outputFolderPath, "input");
            string outputDir = Path.Combine(_outputFolderPath, "output");

            Directory.CreateDirectory(inputDir);
            Directory.CreateDirectory(outputDir);

            string inputFilePath = Path.Combine(inputDir, uniqueFileName);
            string outputFilePath = Path.Combine(outputDir, uniqueFileName);

            try
            {
                using (var stream = new FileStream(inputFilePath, FileMode.Create))
                {
                    await file.CopyToAsync(stream);
                }

                ProcessStartInfo psi = new ProcessStartInfo
                {
                    FileName = _pythonPath,
                    Arguments = $"\"{_scriptPath}\" --input \"{inputFilePath}\" --output \"{outputFilePath}\" --model u2net",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };

                _logger.LogInformation($"Executing command: {_pythonPath} {psi.Arguments}");

                using (Process process = new Process { StartInfo = psi })
                {
                    process.Start();
                    string output = await process.StandardOutput.ReadToEndAsync();
                    string error = await process.StandardError.ReadToEndAsync();
                    await process.WaitForExitAsync();

                    if (process.ExitCode != 0)
                    {
                        _logger.LogError($"Background removal failed: {error}");
                        throw new Exception($"Background removal failed with exit code {process.ExitCode}: {error}");
                    }
                }

                if (!File.Exists(outputFilePath))
                {
                    throw new FileNotFoundException("Output file not found after processing.");
                }

                return outputFilePath;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing image");
                throw new Exception($"Error processing image: {ex.Message}", ex);
            }
        }



        #region  Helpers

        private void EnsureTempDirectoryExists()
        {
            try
            {
                if (!Directory.Exists(_tempDirectory))
                {
                    Directory.CreateDirectory(_tempDirectory);
                    _logger.LogInformation("Created temp directory: {TempDirectory}", _tempDirectory);
                }

                // Check if we have write permissions
                string testFile = Path.Combine(_tempDirectory, "test.txt");
                File.WriteAllText(testFile, "test");
                File.Delete(testFile);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to create or validate temp directory: {TempDirectory}", _tempDirectory);

            }
        }


        #endregion
    }

}