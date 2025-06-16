using System.Security.Claims;
using Microsoft.Extensions.DependencyInjection;

public interface IUserHistoryService
{
    Task RecordNewsReadAsync(string userId, string newsId);
}

public class UserHistoryMiddleware
{
    private readonly RequestDelegate _next;

    public UserHistoryMiddleware(RequestDelegate next)
    {
        _next = next;
    }

    public async Task InvokeAsync(HttpContext context)
    {
        if (context.User.Identity?.IsAuthenticated == true &&
            context.Request.Path.StartsWithSegments("/news", StringComparison.OrdinalIgnoreCase))
        {
            var newsId = context.Request.Query["id"].ToString();
            if (!string.IsNullOrEmpty(newsId))
            {
                var userId = context.User.FindFirst(ClaimTypes.NameIdentifier)?.Value;
                if (userId != null)
                {
                    // Get the service from the request's service scope
                    var historyService = context.RequestServices.GetRequiredService<IUserHistoryService>();
                    await historyService.RecordNewsReadAsync(userId, newsId);
                }
            }
        }

        await _next(context);
    }
}

