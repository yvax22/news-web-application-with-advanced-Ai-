using FND.Models;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;

namespace FND.Services
{
    public class UserHistoryService : IUserHistoryService
    {
        private readonly ILogger<UserHistoryService> _logger;
        private readonly Entity _context;

        public UserHistoryService(ILogger<UserHistoryService> logger, Entity context)
        {
            _logger = logger;
            _context = context;
        }

        public async Task RecordNewsReadAsync(string userId, string newsId)
        {
            try
            {
                // Check if the user has already read this news
                var existingHistory = await _context.UserHistories
                    .FirstOrDefaultAsync(uh => uh.UserId == userId && uh.NewsId == newsId);

                if (existingHistory == null)
                {
                    var userHistory = new UserHistory
                    {
                        UserId = userId,
                        NewsId = newsId,
                        ReadAt = DateTime.UtcNow
                    };

                    await _context.UserHistories.AddAsync(userHistory);
                    await _context.SaveChangesAsync();
                    
                    _logger.LogInformation("Recorded news read for user {UserId} and news {NewsId}", userId, newsId);
                }
                else
                {
                    // Update the read time if the news was read before
                    existingHistory.ReadAt = DateTime.UtcNow;
                    await _context.SaveChangesAsync();
                    
                    _logger.LogInformation("Updated read time for user {UserId} and news {NewsId}", userId, newsId);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error recording news read for user {UserId} and news {NewsId}", userId, newsId);
                throw;
            }
        }
    }
} 