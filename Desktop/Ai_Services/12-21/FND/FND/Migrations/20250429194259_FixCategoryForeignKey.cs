using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace FND.Migrations
{
    public partial class FixCategoryForeignKey : Migration
    {
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            //migrationBuilder.DropForeignKey(
            //    name: "FK_News_Categories_CategoryId",
             //   table: "News");

            migrationBuilder.RenameColumn(
                name: "CategoryId",
                table: "News",
                newName: "CategoryId");

            migrationBuilder.RenameIndex(
                name: "IX_News_Category_Id",
                table: "News",
                newName: "IX_News_CategoryId");

            //migrationBuilder.RenameColumn(
            //    name: "CategoryId",
            //    table: "Categories",
              //  newName: "Category_Id");

            migrationBuilder.AlterColumn<int>(
                name: "CategoryId",
                table: "News",
                type: "int",
                nullable: false,
                defaultValue: 0,
                oldClrType: typeof(int),
                oldType: "int",
                oldNullable: true);

            migrationBuilder.AddForeignKey(
                name: "FK_News_Categories_CategoryId",
                table: "News",
                column: "CategoryId",
                principalTable: "Categories",
                principalColumn: "Id",
                onDelete: ReferentialAction.Cascade);
        }

        protected override void Down(MigrationBuilder migrationBuilder)
        {
           // migrationBuilder.DropForeignKey(
           //     name: "FK_News_Categories_CategoryId",
           //     table: "News");

            migrationBuilder.RenameColumn(
                name: "CategoryId",
                table: "News",
                newName: "Category_Id");

            migrationBuilder.RenameIndex(
                name: "IX_News_CategoryId",
                table: "News",
                newName: "IX_News_Category_Id");

            migrationBuilder.RenameColumn(
                name: "Category_Id",
                table: "Categories",
                newName: "CategoryId");

            migrationBuilder.AlterColumn<int>(
                name: "Category_Id",
                table: "News",
                type: "int",
                nullable: true,
                oldClrType: typeof(int),
                oldType: "int");

            migrationBuilder.AddForeignKey(
                name: "FK_News_Categories_Category_Id",
                table: "News",
                column: "Category_Id",
                principalTable: "Categories",
                principalColumn: "Id");
        }
    }
}
