using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace FND.Migrations
{
    public partial class FixCategoryColumnProperly : Migration
    {
        protected override void Up(MigrationBuilder migrationBuilder)
        {
         //   migrationBuilder.RenameColumn(
           //     name: "Category_Id",
             //   table: "Categories",
               // newName: "CategoryId");
        }

        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.RenameColumn(
                name: "CategoryId",
                table: "Categories",
                newName: "Category_Id");
        }
    }
}
