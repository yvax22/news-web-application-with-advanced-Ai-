using Microsoft.AspNetCore.Identity.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore;

namespace FND.Models
{
    public class Entity : IdentityDbContext<ApplicationUser>
    {
        public Entity(DbContextOptions<Entity> options) : base(options) { }

        public DbSet<News> News { get; set; }
        public DbSet<Category> Categories { get; set; }
        public DbSet<Comment> Comments { get; set; }
        public DbSet<Like> Likes { get; set; }
        public DbSet<UserHistory> UserHistories { get; set; }
        public DbSet<ChatHistory> ChatHistories { get; set; }

        protected override void OnModelCreating(ModelBuilder builder)
        {
            base.OnModelCreating(builder);

            builder.Entity<ChatHistory>().ToTable("ChatHistory");

            builder.Entity<UserHistory>()
                .HasOne(uh => uh.User)
                .WithMany()
                .HasForeignKey(uh => uh.UserId)
                .OnDelete(DeleteBehavior.Cascade);
        }

        protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
        {
            if (!optionsBuilder.IsConfigured)
            {
                optionsBuilder.UseSqlServer("Data Source=. ; Initial Catalog=FND-Yasseen ; Integrated Security=True;Encrypt=True ; Trust Server Certificate=True");
            }
            base.OnConfiguring(optionsBuilder);
        }
    }
}
