using FND.Models;
using Microsoft.AspNetCore.Identity;
using Microsoft.EntityFrameworkCore;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc.Authorization;
using System.Diagnostics;
using System.Net.NetworkInformation;


var builder = WebApplication.CreateBuilder(args);



// Add services to the container.
builder.Services.AddControllersWithViews(options =>
{
    var policy = new AuthorizationPolicyBuilder()
                     .RequireAuthenticatedUser()
                     .Build();
    options.Filters.Add(new AuthorizeFilter(policy));
});

// Database configuration
builder.Services.AddDbContext<Entity>(optionBuilder =>
{
    optionBuilder.UseSqlServer("Data Source=.;Initial Catalog=FND-Yasseen;Integrated Security=True");
});


builder.Services.AddHttpClient<RagService>();

builder.Services.AddIdentity<ApplicationUser, IdentityRole>(options =>
{
    options.Password.RequireDigit = true;
})
.AddEntityFrameworkStores<Entity>()
.AddDefaultTokenProviders();


builder.Services.ConfigureApplicationCookie(options =>
{
    options.LoginPath = "/Account/Login";
    options.LogoutPath = "/Account/Logout";
    options.AccessDeniedPath = "/Account/AccessDenied";
});


// HttpClient configuration
builder.Services.AddHttpClient<RecommendationService>(client =>
{
    client.BaseAddress = new Uri(builder.Configuration["PythonService:Url"] ?? "http://localhost:8500");
});

// User History Service
builder.Services.AddScoped<IUserHistoryService, FND.Services.UserHistoryService>();

// Build the application
var app = builder.Build();

// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Home/Error");
    app.UseHsts();
}

app.UseHttpsRedirection();
app.UseStaticFiles();
app.UseRouting();

app.UseAuthentication();

app.UseAuthorization();

app.MapControllerRoute(
    name: "default",
    pattern: "{controller=News}/{action=AllNews}/{id?}");



app.UseMiddleware<UserHistoryMiddleware>();

// تشغيل ملف main.py عند بدء التشغيل باستخدام مسار مطلق
try
{
    var startInfo = new ProcessStartInfo
    {
        FileName = "python",
        Arguments = "\"C:\\Users\\yasee\\Desktop\\Ai_Services\\Ai_Service\\main.py\"",
        UseShellExecute = false,
        RedirectStandardOutput = true,
        RedirectStandardError = true,
        CreateNoWindow = true
    };

    var process = new Process();
    process.StartInfo = startInfo;
    process.OutputDataReceived += (sender, e) => {
        if (!string.IsNullOrWhiteSpace(e.Data))
            Console.WriteLine($"[PYTHON] {e.Data}");
    };
    process.ErrorDataReceived += (sender, e) => {
        if (!string.IsNullOrWhiteSpace(e.Data))
            Console.WriteLine($"[PYTHON ERROR] {e.Data}");
    };

    process.Start();
    process.BeginOutputReadLine();
    process.BeginErrorReadLine();
}
catch (Exception ex)
{
    Console.WriteLine($"❌ فشل تشغيل main.py: {ex.Message}");
}

app.Run();
