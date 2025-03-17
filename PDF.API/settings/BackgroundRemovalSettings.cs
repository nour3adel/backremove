
namespace PDF.API.settings
{
   public class BackgroundRemovalSettings
    {
        /// <summary>
        /// Path to the Python executable
        /// </summary>
        public string PythonPath { get; set; } = "python";
        
        /// <summary>
        /// Path to the background removal script
        /// </summary>
        public string ScriptPath { get; set; }
        
        /// <summary>
        /// Directory for temporary files
        /// </summary>
        public string TempDirectory { get; set; }
        
        /// <summary>
        /// Timeout in milliseconds for the Python script execution
        /// </summary>
        public int ScriptTimeoutMs { get; set; } = 60000;
        
        public BackgroundRemovalSettings()
        {
            // Default values
            ScriptPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "background_remover.py");
            TempDirectory = Path.Combine(Path.GetTempPath(), "BackgroundRemovalService");
        }
    }
}