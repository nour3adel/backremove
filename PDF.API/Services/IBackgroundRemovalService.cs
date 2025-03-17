

namespace PDF.API.Services
{
    public interface IBackgroundRemovalService
    {
         Task<string> RemoveBackgroundAsync(IFormFile file);
    }
}